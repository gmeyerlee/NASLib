from nasbench import api
import logging
import torch

from naslib.defaults.statistics_evaluator import StatisticsEvaluator

from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    DartsSearchSpace,
    NasBenchNLPSearchSpace,
    TransBench101SearchSpace
)

from naslib.utils import utils, setup_logger, get_dataset_api


config = utils.get_config_from_args(config_type="statistics")
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)

supported_search_spaces = {
    "nasbench101": NasBench101SearchSpace(),
    "nasbench201": NasBench201SearchSpace(),
    "darts": DartsSearchSpace(),
    "nlp": NasBenchNLPSearchSpace(),
    'transbench101': TransBench101SearchSpace()
}
#    'transbench101_micro': TransBench101SearchSpace('micro')
#    'transbench101_macro': TransBench101SearchSpace('micro'),
#    
#}

"""
If the API did not evaluate *all* architectures in the search space, 
set load_labeled=True
"""
load_labeled = True if config.search_space in ["darts", "nlp"] else False
dataset_api = get_dataset_api(config.search_space, config.dataset)

# initialize the search space
search_space = supported_search_spaces[config.search_space]

# initialize the StatisticsEvaluator class
statistics_evaluator = StatisticsEvaluator(config=config)
statistics_evaluator.adapt_search_space(
    search_space, load_labeled=load_labeled, dataset_api=dataset_api
)

# evaluate the statistics
statistics_evaluator.evaluate()
