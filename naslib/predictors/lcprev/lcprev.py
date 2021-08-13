"""
This code is mostly from https://github.com/akshayc11/Spearmint/tree/elc
which is authored by Akshay Chandrashekaran

It is an implementation of
Speeding up Hyper-parameter Optimization by Extrapolation of Learning Curves 
using Previous Builds, Chandrashekaran and Lane, 2017
"""

import numpy as np

from naslib.predictors.predictor import Predictor
from naslib.predictors.lcprev.prevonlyterminationcriterion import \
ConservativeTerminationCriterion

class LCPrevPredictor(Predictor):
    
    def __init__(self, metric=None):
        self.metric = metric

    def fit(self, xtrain, ytrain, info, learn_hyper=True):
        self.train_lcs = np.array([np.array(inf['lc']) / 100 for inf in info])

    def query(self, xtest, info):

        test_lcs = np.array([np.array(inf['lc']) / 100 for inf in info])
        trained_epochs = len(info[0]['lc'])

        # Todo: final_epoch and default_guess should be added as properties
        # of a search space, so that this code doesn't need to be repeated
        if self.ss_type == 'nasbench201':
            final_epoch = 200
            default_guess = 85.0
        elif self.ss_type == 'darts':
            final_epoch = 98
            default_guess = 93.0
        elif self.ss_type == 'nlp':
            final_epoch = 50
            default_guess = 94.83
        else:
            raise NotImplementedError()

        predictions = []
        for i in range(len(xtest)):

            lc = test_lcs[i]
            try:

                """
                Note: this code was written not just to predict,
                but to give a probability that it will hit a certain
                threshold. In NASLib, we only need the former. So the
                parameters like 'predictive_std_threshold', etc,
                don't matter.
                """
                term_crit = ConservativeTerminationCriterion(
                    lc, final_epoch,
                    prob_x_greater_type='posterior_prob_x_greater_than',
                    y_prev_list=self.train_lcs,
                    predictive_std_threshold=.005,
                    min_y_prev=2,
                    recency_weighting=True,
                    monotonicity_condition=False)

                result = term_crit.run(default_guess, threshold=.05)
                prediction = result['predictive_mean'] * 100

            except AssertionError:
                # catch AssertionError in _split_theta method
                print('caught AssertionError running model')
                prediction = np.nan

            if np.isnan(prediction) or not np.isfinite(prediction):
                print('nan or finite')
                prediction = default_guess + np.random.rand()
            predictions.append(prediction)

        predictions = np.array(predictions)
        return predictions            

    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query.
        """
        reqs = {'requires_partial_lc':True, 
                'metric': self.metric,
                'requires_hyperparameters':False,
                'hyperparams':None,
                'unlabeled':False, 
                'unlabeled_factor':0
               }
        return reqs