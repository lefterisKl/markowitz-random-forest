
import numpy as np
import pandas as pd
from tree_weighting.TreeWeightingMethod import TreeWeightingMethod
from pypfopt.risk_models import CovarianceShrinkage
from copy import deepcopy
from sklearn.metrics import average_precision_score,f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score ,mean_absolute_percentage_error
import sklearn.ensemble._forest as forest_utils


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



class IndependentVarianceForest(TreeWeightingMethod):
    #the MRF parameters max_rate,gamma,risk_aversion are ignored and  currently exist only for code compatiblity
    def __init__(self,rate,max_rate,gamma,risk_aversion):
        super().__init__()
        self.rate = rate
        self.max_rate = max_rate
        if isinstance(gamma,tuple):
            self.gamma = gamma[0]
        else:
            self.gamma = gamma
        self.risk_aversion = risk_aversion
        self.weights = []
        self.masking = []

    def estimate_weights_clf(self,Xf_train,yf_train,Xf_val,yf_val,oob_estimate=False,rates=False):
        trees = self.model.estimators_
        tprobs = np.transpose(np.array([tree.predict_proba(Xf_val)[:,1] for tree in trees]))
        if oob_estimate:
            ib,oob = get_ib_oob_samples(self.model,len(yf_train))
            tprobs_train = np.transpose(np.array([tree.predict_proba(Xf_train)[:,1] for tree in trees]))
            mu_ap = np.transpose(np.array([ average_precision_score(
            np.concatenate([yf_val,yf_train[oob[j]]],axis=0),
            np.concatenate([tprobs[:,j],tprobs_train[oob[j],j]],axis=0)) for j in range(tprobs.shape[1])]))
        else:
            raise NotImplementedError
        mu_choice = mu_ap
        accuracy = lambda y,prob: prob**(y) * (1-prob)**(1-y)
        Q = np.transpose(np.array([ accuracy(np.concatenate([yf_val,yf_train],axis=0),
            np.concatenate([tprobs[:,j],tprobs_train[:,j]],axis=0)) for j in range(tprobs.shape[1])]))
        Q = np.clip(Q,a_min=0.00001,a_max=0.99999)
        Q = Q-0.5
        vars = np.var(Q, axis=0, ddof=1) 
        scores = [ mu_choice[i] / vars[i] for i in range(mu_choice.shape[0])]
        scores = np.array(scores) / np.sum(scores)
        self.weights = scores
        return self.weights
    
    def estimate_weights_clf_multi(self,Xf_train,yf_train,Xf_val,yf_val,oob_estimate=False,rates=False):
        trees = self.model.estimators_
        preds = np.transpose(np.array([tree.predict(Xf_val) for tree in trees])) #np.transpose()
        tprobs = np.array([tree.predict_proba(Xf_val) for tree in trees]) #np.transpose()
        tprobs = tprobs.transpose(1, 0, 2)
        # Create an array of row indices to pair with y for selection
        # Use advanced indexing to select the desired elements
        if oob_estimate:
            ib,oob = get_ib_oob_samples(self.model,len(yf_train))
            #tprobs_train = np.transpose(np.array([tree.predict_proba(Xf_train)[:,1] for tree in trees]))
            preds_train = np.transpose(np.array([tree.predict(Xf_train) for tree in trees]))
            tprobs_train = np.array([tree.predict_proba(Xf_train) for tree in trees]) #np.transpose()
            tprobs_train = tprobs_train.transpose(1, 0, 2)
            mu_f1_macro = [f1_score(np.concatenate([yf_val,yf_train[oob[j]]],axis=0),
                np.concatenate([preds[:,j],preds_train[oob[j],j]],axis=0),
                average="macro") for j in range(preds.shape[1])]
        else:
            raise NotImplementedError
        row_indices = np.arange(tprobs.shape[0])
        scores_val = tprobs[row_indices, :, yf_val]
        row_indices_train = np.arange(tprobs_train.shape[0])
        scores_train = tprobs_train[row_indices_train, :, yf_train]
        all_scores = np.concatenate([scores_val,scores_train],axis=0)
        all_tprobs = np.concatenate([tprobs,tprobs_train],axis=0)
        all_preds = np.concatenate([preds,preds_train],axis=0)
        y_all = np.concatenate([yf_val,yf_train],axis=0)
        Q = all_scores
        mu_choice = mu_f1_macro
        classes = pd.Series(y_all).drop_duplicates().sort_values().tolist()
        cond_S = np.zeros((tprobs.shape[1],tprobs.shape[1]))
        for i in range(Q.shape[1]):
            j=i
            coefs = []
            for k in classes:
                idx_false_ik  = (y_all==k) & (all_preds[:,i]!=y_all)
                idx_false_jk  = (y_all==k) & (all_preds[:,j]!=y_all)
                idx_false = idx_false_ik | idx_false_jk
                cor = np.cov( all_tprobs[idx_false,i,:].reshape(-1), all_tprobs[idx_false,j,:].reshape(-1))[0,1]
                coefs.append(cor)
            cond_S[i,j] = np.nanmean( np.array(coefs))
        #Due to this definition of risk matrix, the method depends only the the performance vector
        scores = [ mu_choice[i] / (cond_S[i,i]+0.00000000000000000001) for i in range(len(mu_choice))]
        scores = np.array(scores) / np.sum(scores)   
        self.weights = scores
        return self.weights


    def estimate_weights_reg(self,Xf_train,yf_train,Xf_val,yf_val,oob_estimate=False,rates=False):
        trees = self.model.estimators_
        preds = np.transpose(np.array([tree.predict(Xf_val) for tree in trees]))
        if oob_estimate:
            ib,oob = get_ib_oob_samples(self.model,len(yf_train))
            preds_train = np.transpose(np.array([tree.predict(Xf_train) for tree in trees]))
        mu_r2 = np.transpose(np.array([ r2_score(
        np.concatenate([yf_val,yf_train[oob[j]]],axis=0),
        np.concatenate([preds[:,j],preds_train[oob[j],j]],axis=0)) for j in range(preds.shape[1])]))
        mu_negexpmape = np.transpose(np.array([ np.exp(-mean_absolute_percentage_error(
        np.concatenate([yf_val,yf_train[oob[j]]],axis=0),
        np.concatenate([preds[:,j],preds_train[oob[j],j]],axis=0))) for j in range(preds.shape[1])]))
        score = lambda preds,targets: np.array((targets - preds))
        residuals = np.transpose(np.array([ score(np.concatenate([yf_val,yf_train],axis=0), np.concatenate([preds[:,j],preds_train[:,j]],axis=0)) for j in range(preds.shape[1])]))
        mu_choice = mu_negexpmape
        vars = np.var(residuals, axis=0, ddof=1) 
        scores = [ mu_choice[i] / vars[i] for i in range(mu_choice.shape[0])]
        scores = pd.Series(scores).fillna(np.nanmean(scores)).values
        print(scores)
        if np.max(scores)==0:
            scores = np.array(scores.shape[0]*[1.0/scores.shape[0]])
        else:
            scores = np.array(scores) / np.sum(scores)   
        self.weights = scores
        return self.weights
    
    #prediction for binary classification
    def weighted_predict_proba_n(self,X):
        trees = self.model.estimators_
        masking = self.masking
        tprobs = np.transpose(np.array([tree.predict_proba(X)[:,mask] for tree,mask in zip(trees,masking)]))
        weighted_predictions = np.transpose(np.array([ self.weights[j]*tprobs[:,j] for j in range(self.model.n_estimators)]))
        preds=weighted_predictions.sum(axis=1)
        return preds
    
    #prediction for multi-class classification
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







   
 





