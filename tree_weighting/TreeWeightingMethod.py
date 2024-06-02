from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_absolute_error as mae


class TreeWeightingMethod():
    def __init__(self) -> None:
        pass

    def set_base_model(self,model)->None:
        self.model = model
        self.weights=[ 1.0/self.model.n_estimators for i in range(self.model.n_estimators)]

    @abstractmethod
    def estimate_weights(self):
        pass


    @abstractmethod
    def weighted_prediction(self):
        pass

    def estimate_weights_mae(self,X,y):
        trees = self.model.estimators_
        scores = [ 1/(1+mae(tree.predict(X),y)) for tree in trees]
        self.weights  = scores / np.sum(scores)
        return self.weights
    


    def weighted_predict_proba(self,X):
        trees = self.model.estimators_
        tprobs = np.transpose(np.array([tree.predict_proba(X)[:,1] for tree in trees]))
        weighted_predictions = np.transpose(np.array([ self.weights[j]*tprobs[:,j] for j in range(self.model.n_estimators)]))
        preds=weighted_predictions.sum(axis=1)
        return preds
    
    def weighted_predict(self,X):
        trees = self.model.estimators_
        tpreds = np.transpose(np.array([tree.predict(X) for tree in trees]))
        weighted_predictions = np.transpose(np.array([ self.weights[j]*tpreds[:,j] for j in range(self.model.n_estimators)]))
        preds=weighted_predictions.sum(axis=1)
        return preds

    def weighted_hard_prediction(self,X):
        trees = self.model.estimators_
        tpreds = np.transpose(np.array([tree.predict(X) for tree in trees]))
        weighted_predictions = np.transpose(np.array([ self.weights[j]*tpreds[:,j] for j in range(self.model.n_estimators)]))
        preds=weighted_predictions.sum(axis=1)
        return preds

    def clip_reset(self):
        self.weights= self.weights.clip(min=0.0)
        self.weights = self.weights/self.weights.sum()


    #Prediction function for multiclass classification
    def weighted_predict_multi(self,X):
        trees = self.model.estimators_
        tprobs = np.array([tree.predict_proba(X) for tree in trees])
        tprobs = tprobs.transpose(1, 0, 2)
        weights_reshaped = np.array(self.weights).reshape(1, -1, 1)
        # Broadcast multiply
        tprobs_weighted = tprobs * weights_reshaped
        scores = np.sum(tprobs_weighted, axis=1)
        final_indices = np.argmax(scores, axis=1)
        return final_indices