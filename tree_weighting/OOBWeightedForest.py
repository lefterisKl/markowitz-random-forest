from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tree_weighting.TreeWeightingMethod import TreeWeightingMethod
import sklearn.ensemble._forest as forest_utils
from sklearn.metrics import mean_absolute_error as mae



class OOBWeightedForest(TreeWeightingMethod):
    def __init__(self):
        super().__init__()
        self.weights = []


    def estimate_weights_clf(self,X,y):
        def get_ib_oob_samples(rf,n_samples):
            n_samples_bootstrap = forest_utils._get_n_samples_bootstrap(
                n_samples, rf.max_samples
            )
            unsampled_indices_trees = []
            sampled_indices_trees = []
            for estimator in rf.estimators_:
                unsampled_indices = forest_utils._generate_unsampled_indices(
                    estimator.random_state, n_samples, n_samples_bootstrap)
                unsampled_indices_trees.append(unsampled_indices)
                sampled_indices = forest_utils._generate_sample_indices(
                    estimator.random_state, n_samples, n_samples_bootstrap)
                sampled_indices_trees.append(sampled_indices)
            return sampled_indices_trees, unsampled_indices_trees 
        _,oob = get_ib_oob_samples(self.model,len(y))
        oob_scores = [accuracy_score(self.model.estimators_[i].predict(X[oob[i]]), y[oob[i]]) for i in range(len(oob))]
        oob_scores = np.array(oob_scores) / np.array(oob_scores).sum()
        self.weights = oob_scores
        return self.weights



    def estimate_weights_reg(self,X,y):
        def get_ib_oob_samples(rf,n_samples):
            n_samples_bootstrap = forest_utils._get_n_samples_bootstrap(
                n_samples, rf.max_samples
            )
            unsampled_indices_trees = []
            sampled_indices_trees = []
            for estimator in rf.estimators_:
                unsampled_indices = forest_utils._generate_unsampled_indices(
                    estimator.random_state, n_samples, n_samples_bootstrap)
                unsampled_indices_trees.append(unsampled_indices)
                sampled_indices = forest_utils._generate_sample_indices(
                    estimator.random_state, n_samples, n_samples_bootstrap)
                sampled_indices_trees.append(sampled_indices)
            return sampled_indices_trees, unsampled_indices_trees 
        _,oob = get_ib_oob_samples(self.model,len(y))

        oob_scores = [1/(1+mae(self.model.estimators_[i].predict(X[oob[i]]), y[oob[i]])) for i in range(len(oob))]
        self.weights  = oob_scores / np.sum(oob_scores)
        return self.weights


   
 








