import numpy as np
import torch
import logging
import gc

from naslib.predictors.predictor import Predictor
from naslib.search_spaces.core.query_metrics import Metric

logger = logging.getLogger(__name__)

class ZeroCostBaseline(Predictor):

    def __init__(self, method_type):
        self.method_type = method_type

    def query(self, xtest, info):
        """
        info: a list of dictionaries of size one whose key is method_type.
        Return the value in the dictionary for each architecture.
        """
        return np.array([inf[self.method_type] for inf in info])
        
    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query.
        
        Note: currently defaults/predictor_evaluator.py only looks for 
        hyperparameters if requires_partial_lc is true. Therefore, we 
        need to set it to true even though ZeroCostBaseline does not use 
        LC's. TODO: fix this
        """
        reqs = {'requires_partial_lc':True, 
                'metric': Metric.VAL_ACCURACY,
                'requires_hyperparameters':True,
                'hyperparams':[self.method_type],
                'unlabeled':False
               }
        return reqs