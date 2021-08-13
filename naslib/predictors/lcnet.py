# This is an implementation of the LCNet model from the paper:
# Klein et al. (2017), Learning Curve Prediction with Bayesian Neural Network

import numpy as np
from pybnn.lcnet import LCNet

from naslib.search_spaces.core.query_metrics import Metric
from naslib.predictors.lcsvr import SVR_Estimator


class LCNetPredictor(SVR_Estimator):
    def __init__(self, metric=Metric.VAL_ACCURACY, all_curve=True):
        self.all_curve = all_curve
        self.name = 'LCNet'
        self.metric=metric
        self.require_hyperparameters = False


    def fit(self, xtrain, ytrain, info, learn_hyper=True):
        
        # remove the later epochs of the train learning curves
        # Todo: it would be better to do this inside predictor_evaluator.
        for info_dict in info:
            lc_related_keys = [key for key in info_dict.keys() if 'lc' in key]
            for lc_key in lc_related_keys:
                info_dict[lc_key] = info_dict[lc_key][:info_dict['fidelity']]
        
        # prepare training data
        xtrain_data = self.prepare_data(info)
        y_train = np.array(ytrain)

        best_model = LCNet()
        best_model.train(
            xtrain_data, y_train,
            num_steps=100,
            num_burn_in_steps=10,
            keep_every=5,
            lr=1e-2,
            verbose=True
        )
        self.best_model = best_model


    def query(self, xtest, info):
        data = self.prepare_data(info)
        pred_on_test_set, _ = self.best_model.predict(data)
        return pred_on_test_set

