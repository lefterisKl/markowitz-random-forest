from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tree_weighting.TreeWeightingMethod import TreeWeightingMethod
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score


class ScoreWeightedForest(TreeWeightingMethod):
    def __init__(self):
        super().__init__()
        scorer = None
        self.weights = []

    def estimate_weights_clf(self,X,y):
        trees = self.model.estimators_
        #scores = [ tree.score(X,y) for tree in trees]
        scores = [accuracy_score(tree.predict(X),y) for tree in trees]
        self.weights  = scores / np.sum(scores)
        return self.weights
    

    def estimate_weights_reg(self,X,y):
        trees = self.model.estimators_
        scores = [ 1/( mae(tree.predict(X),y)+np.finfo(np.float64).eps) for tree in trees]
        self.weights  = scores / np.sum(scores)
        return self.weights





   
 





