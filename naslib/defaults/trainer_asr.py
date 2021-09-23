# Define a pure ASR trainer


from .trainer import *
import torch_edit_distance as ed

logger = logging.getLogger(__name__)

def set_time_limit(loader, time_limit):
    db = loader.dataset
    sampler = loader.sampler
    sampler.indices = db.get_indices_shorter_than(time_limit)


class TrainerASR(Trainer):

    def __init__(self, optimizer, config, lightweight_output=False):
        """
        Trainer tailered for ASR datsaet with minimum changes

        Args:
            optimizer: A NASLib optimizer
            config (AttrDict): The configuration loaded from a yaml file, e.g
                via  `utils.get_config_from_args()`

        """
        super().__init__(optimizer, config, lightweight_output)

        # Only tracks training loss, validation loss and perplexity 
        self.train_loss = utils.AverageMeter()
        self.val_per = utils.AverageMeter()
        self.val_loss = utils.AverageMeter()
        assert hasattr(optimizer, 'step_asr'), f'Optimizer {optimizer} does not have step_asr() function! Please implement before using TrainerASR for One-shot methods'

        n_parameters = optimizer.get_model_size()
        logger.info("param size = %fMB", n_parameters)
        self.errors_dict = utils.AttrDict(
            {
                "train_loss": [],
                "valid_per": [],
                "valid_loss": [],
                "test_per": [],
                "test_loss": [],
                "runtime": [],
                "train_time": [],
                "arch_eval": [],
                "params": n_parameters,
            }
        )

    def search(self, resume_from="", summary_writer=None, after_epoch: Callable[[int], None]=None, report_incumbent=True):
        """
        Start the architecture search. Tailered for 

        Generates a json file with training statistics.

        Args:
            resume_from (str): Checkpoint file to resume from. If not given then
                train from scratch.
        """
        logger.info("Start training")

        np.random.seed(self.config.search.seed)
        torch.manual_seed(self.config.search.seed)

        self.optimizer.before_training()
        checkpoint_freq = self.config.search.checkpoint_freq
        if self.optimizer.using_step_function:
            self.scheduler = self.build_search_scheduler(
                self.optimizer.op_optimizer, self.config
            )

            start_epoch = self._setup_checkpointers(
                resume_from, period=checkpoint_freq, scheduler=self.scheduler
            )
        else:
            start_epoch = self._setup_checkpointers(resume_from, period=checkpoint_freq)

        if self.optimizer.using_step_function:
            self.encoder, self.decoder, self.train_queue, self.valid_queue, self.test_queue = self.build_search_dataloaders(
                self.config
            )

        warmup_limits = [1.0, 1.0, 2.0, 2.0]
        warmup = 0
        
        def data_to_gpu(inputs):
            (audio, audio_len), (targets, targets_len) = inputs
            audio = audio.to(device=self.device)
            audio_len = audio_len.to(device=self.device)
            targets = targets.to(device=self.device)
            targets_len = targets_len.to(device=self.device)
            return (audio, audio_len), (targets, targets_len)
        
        for e in range(start_epoch, self.epochs):

            if warmup < len(warmup_limits):
                set_time_limit(self.train_queue, warmup_limits[warmup])
            else:
                set_time_limit(self.train_queue, None)

            start_time = time.time()
            self.optimizer.new_epoch(e)

            if self.optimizer.using_step_function:
                self.optimizer.graph.train()
                import ipdb; ipdb.set_trace()
                for step, data_train in enumerate(self.train_queue):
                    
                    """ 
                    Change this part accordingly for the new trainer...
                    revise data loader accordingly
                    remember to use correct loss function and other stuff

                    """
                    data_train = data_to_gpu(data_train)
                    data_val = next(iter(self.valid_queue))
                    data_val = data_to_gpu(data_val)

                    train_loss, *_ = self.optimizer.step_asr(data_train, data_val, training=True)
                    log_every_n_seconds(
                        logging.INFO,
                        f'{"Warmup e" if warmup < len(warmup_limits) else "E"}' +
                        "poch {}-{}, Train loss: {:.5f}, learning rate: {}".format(
                            warmup if warmup < len(warmup_limits) else e, step, train_loss, self.scheduler.get_last_lr()
                        ),
                        n=5,
                    )

                    if torch.cuda.is_available():
                        log_first_n(
                            logging.INFO,
                            "cuda consumption\n {}".format(torch.cuda.memory_summary()),
                            n=3,
                        )

                    self.train_loss.update(float(train_loss.item()))
                    
                
                # update warmup information
                if warmup < len(warmup_limits):
                    warmup += 1
                    e = 0
                else:
                    self.val_loss.reset()
                    self.val_per.reset()
                    self.optimizer.graph.eval()
                    for val_input in self.valid_queue:
                        loss, logits, logits_len = self.step_asr(val_input, training=False)
                        per = self.decode(logits, logits_len, val_input)
                        self.val_loss.update(loss.item())
                        self.val_per.update(per.item())

                    val_loss = self.val_loss.avg
                    val_per = self.val_per.avg

                    logging.info(f'Epoch {e+1}: average val loss: {val_loss:.4f}, average val per: {val_per:.4f}')
                    if e >= 5: # ignore epochs with time limits
                        self.scheduler.step()
            
                # End of Epoch loop
                end_time = time.time()
                self.errors_dict.train_loss.append(self.train_loss.avg)
                self.errors_dict.valid_per.append(self.val_per.avg)
                self.errors_dict.valid_loss.append(self.val_loss.avg)
                self.errors_dict.runtime.append(end_time - start_time)
            else:
                end_time = time.time()
                # TODO: nasbench101 does not have train_loss, valid_loss, test_loss implemented, so this is a quick fix for now
                # train_per, train_loss, valid_per, valid_loss, test_per, test_loss = self.optimizer.train_statistics()
                (
                    train_per,
                    valid_per,
                    test_per,
                    train_time,
                ) = self.optimizer.train_statistics(report_incumbent)
                train_loss, valid_loss, test_loss = -1, -1, -1

                self.errors_dict.train_per.append(train_per)
                self.errors_dict.train_loss.append(train_loss)
                self.errors_dict.valid_per.append(valid_per)
                self.errors_dict.valid_loss.append(valid_loss)
                self.errors_dict.test_per.append(test_per)
                self.errors_dict.test_loss.append(test_loss)
                self.errors_dict.runtime.append(end_time - start_time)
                self.errors_dict.train_time.append(train_time)
                self.val_per.avg = valid_per

            self.periodic_checkpointer.step(e)

            anytime_results = self.optimizer.test_statistics()
            if anytime_results:
                # record anytime performance
                self.errors_dict.arch_eval.append(anytime_results)
                log_every_n_seconds(
                    logging.INFO,
                    "Epoch {}, Anytime results: {}".format(e, anytime_results),
                    n=5,
                )

            self._log_to_json()

            self._log_and_reset_accuracies(e, summary_writer)

            if after_epoch is not None:
                after_epoch(e)

        self.optimizer.after_training()

        if summary_writer is not None:
            summary_writer.close()

        logger.info("Training finished")

    def decode(self, output, output_len, val_inputs):
        _, (targets, targets_len) = val_inputs
        targets = targets.to(device=self.device, dtype=torch.int)
        targets_len = targets_len.to(device=self.device, dtype=torch.int)

        targets = self.encoder.fold_encoded(targets, 39)

        beams, _, _, beams_len = self.decoder.decode(output, output_len)
        top_beams = beams[:,0].to(device=self.device, dtype=torch.int)
        top_beams_len = beams_len[:,0].to(device=self.device, dtype=torch.int)

        top_beams = self.encoder.fold_encoded(top_beams, 39)

        blank = torch.Tensor([0]).to(device=self.device, dtype=torch.int)
        sep = torch.Tensor([]).to(device=self.device, dtype=torch.int)

        per = ed.compute_wer(top_beams, targets, top_beams_len, targets_len, blank, sep)
        per = per.mean()
        return per
    
    @staticmethod
    def build_search_dataloaders(config):
        encoder, decoder, train_queue, valid_queue, _ = utils.get_train_val_loaders(
            config, mode="train"
        )
        return encoder, decoder, train_queue, valid_queue, None  # test_queue is not used in search currently

    @staticmethod
    def build_eval_dataloaders(config):
        encoder, decoder, train_queue, valid_queue, test_queue = utils.get_train_val_loaders(
            config, mode="val"
        )
        return encoder, decoder, train_queue, valid_queue, test_queue

    # def _prepare_dataloaders(self, config, mode="train"):
    #     """
    #     Prepare train, validation, and test dataloaders with the splits defined
    #     in the config.

    #     Args:
    #         config (AttrDict): config from config file.
    #     """
    #     if mode == 'train':
    #         encoder, train_queue, valid_queue, test_queue = self.build_search_dataloaders(
    #             config, mode
    #         )
    #     elif mode == 'val':
    #         encoder, train_queue, valid_queue, test_queue = self.build_eval_dataloaders(
    #             config, mode
    #         )
    #     self.train_queue = train_queue
    #     self.valid_queue = valid_queue
    #     self.test_queue = test_queue
    #     self.encoder = encoder

    def _store_per(self, inputs, logits, logits_len, mode):
        """Update the perplexity counters"""
        if mode == "train":
            return
        elif mode == "val":
            # n = logits.size(0)
            per = self.decode(logits, logits_len, inputs)
            self.val_per.update(per.item())
        else:
            raise ValueError("Unknown split: {}. Expected either 'train' or 'val'")

    def _log_and_reset_accuracies(self, epoch, writer=None):
        logger.info(
            "Epoch {} done. Train loss: {:.5f}, Validation loss: {:.5f}, per {:.5f}".format(
                epoch,
                self.train_loss.avg,
                self.val_loss.avg,
                self.val_per.avg,
            )
        )

        if writer is not None:
            writer.add_scalar('Train loss', self.train_loss.avg, epoch)
            writer.add_scalar('Validation per', self.val_per.avg, epoch)
            writer.add_scalar('Validation loss', self.val_loss.avg, epoch)

        self.train_loss.reset()
        self.val_loss.reset()
        self.val_per.reset()