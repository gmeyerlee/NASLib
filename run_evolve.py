import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import EvolvingOptimizer
from naslib.search_spaces import NasBench201SearchSpace

from naslib.utils import set_seed, setup_logger, get_config_from_args, get_dataset_api

config = get_config_from_args()  # use --help so see the options
set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)  # default DEBUG is very verbose
logger.info(f"Seed is {config.seed}")

search_space = NasBench201SearchSpace()  # use SimpleCellSearchSpace() for less heavy search

optimizer = EvolvingOptimizer(config, innov_protect=True)
#dataset_api = get_dataset_api(search_space, config.dataset)
optimizer.adapt_search_space(search_space) #, dataset_api=dataset_api)

trainer = Trainer(optimizer, config)
trainer.search()  # Search for an architecture
trainer.evaluate() #dataset_api=dataset_api)  # Evaluate the best architecture
