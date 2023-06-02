import codecs
from naslib.search_spaces.core.graph import Graph
import time
import json
import logging
import os
import copy
import math
import torch
import numpy as np

from fvcore.common.checkpoint import PeriodicCheckpointer

from naslib.search_spaces.core.query_metrics import Metric

from naslib import utils
from naslib.utils.log import log_every_n_seconds, log_first_n
from naslib.search_spaces.nasbench301.conversions import convert_naslib_to_genotype
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_op_indices, convert_op_indices_to_str

from typing import Callable
from .additional_primitives import DropPathWrapper
from .trainer import Trainer

logger = logging.getLogger(__name__)

def is_valid_arch(op_indices: list) -> bool:
            return not ((op_indices[0] == op_indices[1] == op_indices[2] == 1) or
                        (op_indices[2] == op_indices[4] == op_indices[5] == 1))

class TrainerEval(Trainer):
    """
    Trainer subclass which adds additional evaluation options for robust benchmark-base evaluation.
    """

    def __init__(self, optimizer, config, lightweight_output=False):
        """
        Initializes the trainer.

        Args:
            optimizer: A NASLib optimizer
            config (AttrDict): The configuration loaded from a yaml file, e.g
                via  `utils.get_config_from_args()`
        """
        super().__init__(optimizer, config, lightweight_output)
        

    def eval_sharedw(self, resume_from="", sample_size=0, dataset_api=None, eval_portion=0, tr_ep=0):
        """
        Evaluate the shared weights by iteratively evaluating discrete architectures from the search space.
        Args:
            resume_from (str): Checkpoint file to resume from. If not given then
                evaluate with the current one-shot weights.
        """
        logger.info("Start shared-weight evaluation")
        self.optimizer.before_training()
        self._setup_checkpointers(resume_from)

        if sample_size:
            if self.optimizer.graph.get_type() == "nasbench201":
                xtest = [self._sample_nb201() for _ in range(sample_size)]
        else:
            xtest = self.optimizer.graph.get_arch_iterator(dataset_api)
            if self.optimizer.graph.get_type() == "nasbench201":
                xtest = [list(x) for x in xtest if is_valid_arch(list(x))]
        #if self.optimizer.graph.get_type() == "nasbench301":
        #    converter = convert_naslib_to_genotype
        #elif self.optimizer.graph.get_type() == "nasbench201":
        #    converter = convert_naslib_to_op_indices
        #else:
        #    raise NotImplementedError
        #_xtest = [converter(arch) for arch in xtest]
        #archs = np.squeeze(np.array(self(_xtest)))
        _, dataloader, _ = self.build_search_dataloaders(self.config)
        prediction = []
        timing = []
        true_acc = []
        for arch in xtest:
            # we have to iterate through all the architectures in the
            # mini-batch
            self.optimizer.set_alphas_from_path(arch)
            self.val_top1.reset()
            self.val_top5.reset()
            # NOTE: evaluation on the 25k validation data for now. provide a test
            # dataloader to evaluate on the test data
            val_acc, rtime = self._eval_oneshot(dataloader=dataloader, eval_portion=eval_portion)
            prediction.append(val_acc)
            timing.append(rtime)
            result = self._query_nb201(dataset_api, arch)
            true_acc.append(result)
            logger.info(f"{arch}: {val_acc} {result}")

        sharedw_res = {'sharedw_val': prediction, 'eval_time': timing, 'bench_test': true_acc}

        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        if not tr_ep:
            with codecs.open(
                os.path.join(self.config.save, "sharedw_eval.json"), "w", encoding="utf-8"
            ) as file:
                json.dump(sharedw_res, file, separators=(",", ":"))
        else:
            with codecs.open(
                os.path.join(self.config.save, f"sharedw_eval_{tr_ep}.json"), "w", encoding="utf-8"
            ) as file:
                json.dump(sharedw_res, file, separators=(",", ":"))
 
    def eval_archw(self, resume_from="", dataset_api=None, tr_ep=0):
        """
        Evaluate the architecture weights (if present) by iteratively evaluating discrete architectures from the search space.
        Args:
            resume_from (str): Checkpoint file to resume from. If not given then
                evaluate with the current one-shot weights.
        """
        logger.info("Start architecture weight evaluation")
        self.optimizer.before_training()
        self._setup_checkpointers(resume_from)
        bench_acc = []
        a_probs = []
        if self.config.search_space == "nasbench201":
            xtest = self.optimizer.graph.get_arch_iterator(dataset_api)
            archs = [list(x) for x in xtest if is_valid_arch(list(x))]
            for arch in archs:
              result = self._query_nb201(dataset_api, arch)
              bench_acc.append(result)
              a_prob = np.prod([torch.softmax(alpha, dim=-1).detach().cpu()[i] for i, alpha in zip(arch, self.optimizer.architectural_weights)], dtype=np.float64)
              a_probs.append(a_prob)
        archw_res = {'archw_prob': a_probs, 'bench_test': bench_acc}

        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        if not tr_ep:
            with codecs.open(
                os.path.join(self.config.save, "archw_eval.json"), "w", encoding="utf-8"
            ) as file:
                json.dump(archw_res, file, separators=(",", ":"))
        else:
            with codecs.open(
                os.path.join(self.config.save, f"archw_eval_{tr_ep}.json"), "w", encoding="utf-8"
            ) as file:
                json.dump(archw_res, file, separators=(",", ":"))


    def _sample_nb201(self):

        NUM_EDGES = 6
        NUM_OPS = 5
 
        while True:
            op_indices = np.random.randint(NUM_OPS, size=(NUM_EDGES)).tolist()

            if not is_valid_arch(op_indices):
                continue
            return op_indices
    def _query_nb201(self, dataset_api, op_indices):
        arch_str = convert_op_indices_to_str(op_indices)
        dataset = self.config.dataset 

        if dataset in ["cifar10", "cifar10-valid"]:
            query_results = dataset_api["nb201_data"][arch_str]
            # set correct cifar10 dataset
            dataset = "cifar10-valid"
        elif dataset == "cifar100":
            query_results = dataset_api["nb201_data"][arch_str]
        elif dataset == "ImageNet16-120":
            query_results = dataset_api["nb201_data"][arch_str]
        else:
            raise NotImplementedError("Invalid dataset")
 
        return query_results[dataset]["eval_acc1es"][-1]
    
    def _eval_oneshot(self, resume_from="", dataloader=None, eval_portion=0):
        """
        Evaluate the one-shot model on the specified dataset.
        Modified function made to be run repeatedly with different architectures.
        No longer overwrites errors.json

        Args:
            resume_from (str): Checkpoint file to resume from. If not given then
                evaluate with the current one-shot weights.
        """
        #self.optimizer.before_training()
        #self._setup_checkpointers(resume_from)

        loss = torch.nn.CrossEntropyLoss()

        if dataloader is None:
            # load only the validation data
            _, dataloader, _ = self.build_search_dataloaders(self.config) 

        self.optimizer.graph.eval()
        with torch.no_grad():
            start_time = time.time()
            if eval_portion:
                data_iter = iter(dataloader)
                for step in range(math.floor(len(dataloader) * eval_portion)):
                    data_val = next(data_iter)
                    input_val = data_val[0].to(self.device)
                    target_val = data_val[1].to(self.device, non_blocking=True)

                    logits_val = self.optimizer.graph(input_val)

                    self._store_accuracies(logits_val, data_val[1], "val")
            else:
                for step, data_val in enumerate(dataloader):
                    input_val = data_val[0].to(self.device)
                    target_val = data_val[1].to(self.device, non_blocking=True)

                    logits_val = self.optimizer.graph(input_val)

                    self._store_accuracies(logits_val, data_val[1], "val")

            end_time = time.time()

            #self.search_trajectory.valid_acc.append(self.val_top1.avg)
            #self.search_trajectory.valid_loss.append(self.val_loss.avg)
            #self.search_trajectory.runtime.append(end_time - start_time)

            #self._log_to_json()

        return self.val_top1.avg, end_time-start_time

    def evaluate(
        self,
        retrain:bool=True,
        search_model:str="",
        resume_from:str="",
        best_arch:Graph=None,
        dataset_api:object=None,
        metric:Metric=None,
    ):
        """
        Evaluate the final architecture as given from the optimizer.

        If the search space has an interface to a benchmark then query that.
        Otherwise train as defined in the config.

        Args:
            retrain (bool)      : Reset the weights from the architecure search
            search_model (str)  : Path to checkpoint file that was created during search. If not provided,
                                  then try to load 'model_final.pth' from search
            resume_from (str)   : Resume retraining from the given checkpoint file.
            best_arch           : Parsed model you want to directly evaluate and ignore the final model
                                  from the optimizer.
            dataset_api         : Dataset API to use for querying model performance.
            metric              : Metric to query the benchmark for.
        """
        logger.info("Start evaluation")
        if not best_arch:

            if not search_model:
                search_model = os.path.join(
                    self.config.save, "search", "model_final.pth"
                )
            self._setup_checkpointers(search_model)  # required to load the architecture

            best_arch = self.optimizer.get_final_architecture()
        logger.info(f"Final architecture hash: {best_arch.get_hash()}")

        if best_arch.QUERYABLE:
            if metric is None:
                metric = Metric.TEST_ACCURACY
            result = best_arch.query(
                metric=metric, dataset=self.config.dataset, dataset_api=dataset_api
            )
            logger.info("Queried results ({}): {}".format(metric, result))
            return result
        else:
            best_arch.to(self.device)
            if retrain:
                logger.info("Starting retraining from scratch")
                best_arch.reset_weights(inplace=True)

                (
                    self.train_queue,
                    self.valid_queue,
                    self.test_queue,
                ) = self.build_eval_dataloaders(self.config)

                optim = self.build_eval_optimizer(best_arch.parameters(), self.config)
                scheduler = self.build_eval_scheduler(optim, self.config)

                start_epoch = self._setup_checkpointers(
                    resume_from,
                    search=False,
                    period=self.config.evaluation.checkpoint_freq,
                    model=best_arch,  # checkpointables start here
                    optim=optim,
                    scheduler=scheduler,
                )

                grad_clip = self.config.evaluation.grad_clip
                loss = torch.nn.CrossEntropyLoss()

                self.train_top1.reset()
                self.train_top5.reset()
                self.val_top1.reset()
                self.val_top5.reset()

                # Enable drop path
                best_arch.update_edges(
                    update_func=lambda edge: edge.data.set(
                        "op", DropPathWrapper(edge.data.op)
                    ),
                    scope=best_arch.OPTIMIZER_SCOPE,
                    private_edge_data=True,
                )

                # train from scratch
                epochs = self.config.evaluation.epochs
                for e in range(start_epoch, epochs):
                    best_arch.train()

                    if torch.cuda.is_available():
                        log_first_n(
                            logging.INFO,
                            "cuda consumption\n {}".format(torch.cuda.memory_summary()),
                            n=20,
                        )

                    # update drop path probability
                    drop_path_prob = self.config.evaluation.drop_path_prob * e / epochs
                    best_arch.update_edges(
                        update_func=lambda edge: edge.data.set(
                            "drop_path_prob", drop_path_prob
                        ),
                        scope=best_arch.OPTIMIZER_SCOPE,
                        private_edge_data=True,
                    )

                    # Train queue
                    for i, (input_train, target_train) in enumerate(self.train_queue):
                        input_train = input_train.to(self.device)
                        target_train = target_train.to(self.device, non_blocking=True)

                        optim.zero_grad()
                        logits_train = best_arch(input_train)
                        train_loss = loss(logits_train, target_train)
                        if hasattr(
                            best_arch, "auxilary_logits"
                        ):  # darts specific stuff
                            log_first_n(logging.INFO, "Auxiliary is used", n=10)
                            auxiliary_loss = loss(
                                best_arch.auxilary_logits(), target_train
                            )
                            train_loss += (
                                self.config.evaluation.auxiliary_weight * auxiliary_loss
                            )
                        train_loss.backward()
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(
                                best_arch.parameters(), grad_clip
                            )
                        optim.step()

                        self._store_accuracies(logits_train, target_train, "train")
                        log_every_n_seconds(
                            logging.INFO,
                            "Epoch {}-{}, Train loss: {:.5}, learning rate: {}".format(
                                e, i, train_loss, scheduler.get_last_lr()
                            ),
                            n=5,
                        )

                    # Validation queue
                    if self.valid_queue:
                        best_arch.eval()
                        for i, (input_valid, target_valid) in enumerate(
                            self.valid_queue
                        ):

                            input_valid = input_valid.to(self.device).float()
                            target_valid = target_valid.to(self.device).float()

                            # just log the validation accuracy
                            with torch.no_grad():
                                logits_valid = best_arch(input_valid)
                                self._store_accuracies(
                                    logits_valid, target_valid, "val"
                                )

                    scheduler.step()
                    self.periodic_checkpointer.step(e)
                    self._log_and_reset_accuracies(e)

            # Disable drop path
            best_arch.update_edges(
                update_func=lambda edge: edge.data.set(
                    "op", edge.data.op.get_embedded_ops()
                ),
                scope=best_arch.OPTIMIZER_SCOPE,
                private_edge_data=True,
            )

            # measure final test accuracy
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()

            best_arch.eval()

            for i, data_test in enumerate(self.test_queue):
                input_test, target_test = data_test
                input_test = input_test.to(self.device)
                target_test = target_test.to(self.device, non_blocking=True)

                n = input_test.size(0)

                with torch.no_grad():
                    logits = best_arch(input_test)

                    prec1, prec5 = utils.accuracy(logits, target_test, topk=(1, 5))
                    top1.update(prec1.data.item(), n)
                    top5.update(prec5.data.item(), n)

                log_every_n_seconds(
                    logging.INFO,
                    "Inference batch {} of {}.".format(i, len(self.test_queue)),
                    n=5,
                )

            logger.info(
                "Evaluation finished. Test accuracies: top-1 = {:.5}, top-5 = {:.5}".format(
                    top1.avg, top5.avg
                )
            )

            return top1.avg 
