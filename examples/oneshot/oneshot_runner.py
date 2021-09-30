import logging
import sys
import naslib as nl

from naslib.defaults.predictor_evaluator import PredictorEvaluator
from naslib.defaults.trainer import Trainer
from naslib.defaults.trainer_asr import TrainerASR
from naslib.optimizers import Bananas, OneShotNASOptimizer, RandomNASOptimizer, DARTSOptimizer
from naslib.predictors import OneShotPredictor

from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    DartsSearchSpace,
    NasBenchNLPSearchSpace,
    TransBench101SearchSpace,
    NasBenchASRSearchSpace,
)
from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.utils.utils import get_project_root


config = utils.get_config_from_args(config_type="oneshot")

logger = setup_logger(config.save + "/log.log", debug=True)

utils.log_args(config)

supported_optimizers = {
    "bananas": Bananas(config),
    "oneshot": OneShotNASOptimizer(config),
    "rsws": RandomNASOptimizer(config),
    "darts": DARTSOptimizer(config)
}

supported_search_spaces = {
    "nasbench101": NasBench101SearchSpace(),
    "nasbench201": NasBench201SearchSpace(),
    "darts": DartsSearchSpace(),
    'nasbenchnlp': NasBenchNLPSearchSpace(),
    'transbench101': TransBench101SearchSpace(),
    "nasbenchasr": NasBenchASRSearchSpace()
}


# load_labeled = (True if config.search_space == 'darts' else False)
load_labeled = False
dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

# choose trainer
if config.dataset in ['timit']:
    # make sure this is the only calling difference...
    trainer = TrainerASR(optimizer, config, lightweight_output=True)
else:
    trainer = Trainer(optimizer, config, lightweight_output=True)

if config.optimizer == "bananas":
    trainer.search(resume_from="")
    trainer.evaluate(resume_from="", dataset_api=dataset_api)
elif config.optimizer in ["oneshot", "rsws", "darts"]:
    predictor = OneShotPredictor(config, trainer, model_path=config.model_path)

    predictor_evaluator = PredictorEvaluator(predictor, config=config)
    predictor_evaluator.adapt_search_space(
        search_space, load_labeled=load_labeled, dataset_api=dataset_api
    )

    # evaluate the predictor
    predictor_evaluator.evaluate()