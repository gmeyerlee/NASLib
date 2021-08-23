import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import BOHBSimple, DARTSOptimizer, RegularizedEvolution
from naslib.search_spaces import NasBench201SearchSpace, NasBench101SearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args, get_dataset_api

config = get_config_from_args()  # use --help so see the options
print(config.seed)
set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is very verbose

search_space = NasBench201SearchSpace()  # use SimpleCellSearchSpace() for less heavy search

optimizer = BOHBSimple(config)
print(config.search_space)
dataset_api = get_dataset_api(config.search_space, dataset=config.dataset)
optimizer.adapt_search_space(search_space=search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config)
trainer.search()  # Search for an architecture
trainer.evaluate()  # Evaluate the best architecture