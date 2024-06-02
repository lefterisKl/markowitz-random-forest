from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import  precision_score
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
from sklearn.metrics import log_loss
from tree_weighting.TreeWeightingMethod import TreeWeightingMethod
from pypfopt.risk_models import CovarianceShrinkage
from copy import deepcopy
from sklearn.metrics import balanced_accuracy_score,precision_score,recall_score,average_precision_score,f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import roc_auc_score,mean_absolute_percentage_error
import cvxpy as  cp




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



class MarkowitzForest(TreeWeightingMethod):
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

    #MRF weight estimation for binary classification
    def estimate_weights_clf(self,Xf_train,yf_train,Xf_val,yf_val,oob_estimate=False,rates=False):
        trees = self.model.estimators_
        tprobs = np.transpose(np.array([tree.predict_proba(Xf_val)[:,1] for tree in trees]))
        imbalance = dict(pd.Series(yf_val).value_counts())
        if oob_estimate:
            _,oob = get_ib_oob_samples(self.model,len(yf_train))
            tprobs_train = np.transpose(np.array([tree.predict_proba(Xf_train)[:,1] for tree in trees]))
        #Computation of the performance vector for binary classification
        if oob_estimate:
            mu_ap = np.transpose(np.array([ average_precision_score(
            np.concatenate([yf_val,yf_train[oob[j]]],axis=0),
            np.concatenate([tprobs[:,j],tprobs_train[oob[j],j]],axis=0)) for j in range(tprobs.shape[1])]))
        else:
            raise NotImplementedError
        mu_choice = mu_ap
        #Computation of Risk Matrix for binary classification
        accuracy = lambda y,prob: prob**(y) * (1-prob)**(1-y)
        Q = np.transpose(np.array([ accuracy(np.concatenate([yf_val,yf_train],axis=0),
            np.concatenate([tprobs[:,j],tprobs_train[:,j]],axis=0)) for j in range(tprobs.shape[1])]))
        yq = np.concatenate([yf_val,yf_train])
        Q = np.clip(Q,a_min=0.00001,a_max=0.99999)
        Q = Q-0.5
        corr = lambda v1,v2: np.corrcoef( v1, v2)[0,1]
        cond_S = np.zeros((Q.shape[1],Q.shape[1]))
        for i in range(Q.shape[1]):
            for j in range(Q.shape[1]):
                if i < j:
                    idx_TP_i = (Q[:,i] > 0) & (yq==1)
                    idx_TP_j = (Q[:,j] > 0) & (yq==1)
                    idx_FP_i = (Q[:,i] > 0) & (yq==0)
                    idx_FP_j = (Q[:,j] > 0) & (yq==0)
                    idx_FN_i = (Q[:,i] < 0) & (yq==1)
                    idx_FN_j = (Q[:,j] < 0) & (yq==1)
                    idx_TP = idx_TP_i | idx_TP_j
                    idx_FP = idx_FP_i | idx_FP_j
                    idx_FN = idx_FN_i | idx_FN_j
                    vi_FP = Q[idx_FP,i]
                    vj_FP = Q[idx_FP,j]
                    vi_FN = Q[idx_FN,i]
                    vj_FN = Q[idx_FN,j]
                    corr_FP = corr(vi_FP,vj_FP)
                    corr_FN = corr(vi_FN,vj_FN)
                    cond_S[i,j] = (corr_FP + corr_FN)/2.0 
                    cond_S[j,i] = cond_S[i,j] 
        for i in range(cond_S.shape[0]):
            cond_S[i,i] = 1
        #Shrink the risk matrix if it is not positive definite.
        eigenvalues, eigenvectors = np.linalg.eigh(cond_S)
        cond_use = cond_S
        if np.min(eigenvalues)< 0:
            V = eigenvectors
            new_eigenvalues = eigenvalues[:]
            new_eigenvalues[new_eigenvalues < 0.000000001 ] = 0.000000001
            D0 = np.diag(new_eigenvalues)
            V_inv = np.linalg.inv(V)
            cond_S_0 = np.dot(np.dot(V, D0), V_inv)
            cond_use = cond_S_0
        if self.rate is None:
            lower=None
        else:
            lower= self.rate/self.model.n_estimators
        if self.max_rate is None:
            upper=None
        else:
            upper= self.max_rate/self.model.n_estimators
        if rates:
            ef = EfficientFrontier(mu_choice, cond_use, weight_bounds=(lower, upper))
        else:
            ef = EfficientFrontier(mu_choice, cond_use)
        if self.gamma is not None:
            ef.add_objective(objective_functions.L2_reg, gamma=self.gamma)
        #The following function solves the mean variance optimization problem with quadratic programming
        try:
            if self.risk_aversion is not None:
                raw_weights =  ef.max_quadratic_utility(risk_aversion=self.risk_aversion)

            else:
                raw_weights = ef.max_quadratic_utility()
        except Exception as e:
            print(e)
            return self.weights
        cleaned_weights = raw_weights #= ef.clean_weights()
        self.weights = [cleaned_weights[i] for i in range(len(cleaned_weights))]
        return self.weights


    #MRF weight estimation for multi-class classification
    def estimate_weights_clf_multi(self,Xf_train,yf_train,Xf_val,yf_val,oob_estimate=False,rates=False):
        trees = self.model.estimators_
        preds = np.transpose(np.array([tree.predict(Xf_val) for tree in trees])) 
        tprobs = np.array([tree.predict_proba(Xf_val) for tree in trees]) 
        tprobs = tprobs.transpose(1, 0, 2)
        #Estimate the performance vector
        if oob_estimate:
            ib,oob = get_ib_oob_samples(self.model,len(yf_train))
            preds_train = np.transpose(np.array([tree.predict(Xf_train) for tree in trees]))
            tprobs_train = np.array([tree.predict_proba(Xf_train) for tree in trees]) #np.transpose()
            tprobs_train = tprobs_train.transpose(1, 0, 2)
        mu_f1_macro = [f1_score(np.concatenate([yf_val,yf_train[oob[j]]],axis=0),
                        np.concatenate([preds[:,j],preds_train[oob[j],j]],axis=0),
                        average="macro") for j in range(preds.shape[1])]
        mu_choice = mu_f1_macro
        #Estimate the risk matrix
        row_indices = np.arange(tprobs.shape[0])
        scores_val = tprobs[row_indices, :, yf_val]
        row_indices_train = np.arange(tprobs_train.shape[0])
        scores_train = tprobs_train[row_indices_train, :, yf_train]
        all_scores = np.concatenate([scores_val,scores_train],axis=0)
        all_tprobs = np.concatenate([tprobs,tprobs_train],axis=0)
        all_preds = np.concatenate([preds,preds_train],axis=0)
        y_all = np.concatenate([yf_val,yf_train],axis=0)
        Q = all_scores
        classes = pd.Series(y_all).drop_duplicates().sort_values().tolist()
        cond_S = np.zeros((tprobs.shape[1],tprobs.shape[1]))
        for i in range(Q.shape[1]):
            for j in range(Q.shape[1]):
                if i < j:
                    coefs = []
                    for k in classes:
                        idx_false_ik  = (y_all==k) & (all_preds[:,i]!=y_all)
                        idx_false_jk  = (y_all==k) & (all_preds[:,j]!=y_all)
                        idx_false = idx_false_ik | idx_false_jk
                        idx_false_both = idx_false_ik & idx_false_jk
                        cor = np.corrcoef( all_tprobs[idx_false,i,:].reshape(-1), all_tprobs[idx_false,j,:].reshape(-1))[0,1]
                        coefs.append(cor)
                    cond_S[i,j] = np.nanmean( np.array(coefs))
                    cond_S[j,i] = cond_S[i,j]
        for i in range(Q.shape[1]):
            cond_S[i,i]=1
        cond_use = cond_S
        #Shrink the risk matrix if it is not positive definite.
        eigenvalues, eigenvectors = np.linalg.eigh(cond_S)
        if np.min(eigenvalues)< 0:
            shrinking = True
            V = eigenvectors
            new_eigenvalues = eigenvalues[:]
            new_eigenvalues[new_eigenvalues < 0.000000001 ] = 0.000000001
            D0 = np.diag(new_eigenvalues)
            V_inv = np.linalg.inv(V)
            cond_S_0 = np.dot(np.dot(V, D0), V_inv)
            cond_use = cond_S_0
        if self.rate is None:
            lower=None
        else:
            lower= self.rate/self.model.n_estimators
        if self.max_rate is None:
            upper=None
        else:
            upper= self.max_rate/self.model.n_estimators
        if rates:
            ef = EfficientFrontier(mu_choice, cond_use, weight_bounds=(lower, upper))
        else:
            ef = EfficientFrontier(mu_choice, cond_use)
        import cvxpy as  cp
        def L1_norm(w, k=1):
            return k * cp.norm(w, 1)
        def L1_norm(w, k=1):
            return k * cp.norm(w, 1)
        if self.gamma is not None:
            ef.add_objective(objective_functions.L2_reg, gamma=self.gamma)
        #Solve the mean variance optimization problem
        try:
            if self.risk_aversion is not None:
                raw_weights =  ef.max_quadratic_utility(risk_aversion=self.risk_aversion)
            else:
                raw_weights = ef.max_quadratic_utility()
        except Exception as e:
            print(e)
            return self.weights
        cleaned_weights = raw_weights
        self.weights = [cleaned_weights[i] for i in range(len(cleaned_weights))]
        return self.weights


    def estimate_weights_reg(self,Xf_train,yf_train,Xf_val,yf_val,oob_estimate=False,rates=False):
        trees = self.model.estimators_
        preds = np.transpose(np.array([tree.predict(Xf_val) for tree in trees]))
        if oob_estimate:
            ib,oob = get_ib_oob_samples(self.model,len(yf_train))
            preds_train = np.transpose(np.array([tree.predict(Xf_train) for tree in trees]))
        #Estimate the performance vector for regression
        mu_negexpmape = np.transpose(np.array([ np.exp(-mean_absolute_percentage_error(
        np.concatenate([yf_val,yf_train[oob[j]]],axis=0),
        np.concatenate([preds[:,j],preds_train[oob[j],j]],axis=0))) for j in range(preds.shape[1])]))
        mu_choice = mu_negexpmape
        #Estimate the risk matrix for regression
        score = lambda preds,targets: np.array((targets - preds))
        residuals = np.transpose(np.array([ score(np.concatenate([yf_val,yf_train],axis=0), np.concatenate([preds[:,j],preds_train[:,j]],axis=0)) for j in range(preds.shape[1])]))
        Q = residuals
        cond_S = np.zeros((Q.shape[1],Q.shape[1]))
        for i in range(Q.shape[1]):
            for j in range(Q.shape[1]):
                if i <= j:
                    cond_S[i,j] = np.sum(Q[:,i]*Q[:,j]) / np.sqrt(np.sum((Q[:,i]**2)*(Q[:,j]**2)))
                    cond_S[j,i] = cond_S[i,j] 
        cond_use = cond_S
        if self.rate is None:
            lower=None
        else:
            lower= self.rate/self.model.n_estimators
        if self.max_rate is None:
            upper=None
        else:
            upper= self.max_rate/self.model.n_estimators
        if rates:
            ef = EfficientFrontier(mu_choice, cond_use, weight_bounds=(lower, upper))
        else:
            ef = EfficientFrontier(mu_choice, cond_use)
        import cvxpy as  cp
        def L1_norm(w, k=1):
            return k * cp.norm(w, 1)
        def L1_norm(w, k=1):
            return k * cp.norm(w, 1)

        if self.gamma is not None:
            ef.add_objective(objective_functions.L2_reg, gamma=self.gamma)
        #Solve the mean-variance weight optimization problem for regression
        try:
            if self.risk_aversion is not None:
                raw_weights =  ef.max_quadratic_utility(risk_aversion=self.risk_aversion)
            else:
                raw_weights = ef.max_quadratic_utility()
        except Exception as e:
            print(e)
            return self.weights,mu_choice,cond_use
        cleaned_weights = raw_weights 
        self.weights = [cleaned_weights[i] for i in range(len(cleaned_weights))]
        return self.weights,mu_choice, cond_use
    

    #Prediction function for binary classification
    #Note: masking is used to invert the probabilities of trees that have negative weights, into 1-p.
    #However, the originally proposed MRF always set the constraint w>0 anyway, so this feature is not relevant
    def weighted_predict_proba_n(self,X):
        trees = self.model.estimators_
        masking = self.masking
        tprobs = np.transpose(np.array([tree.predict_proba(X)[:,mask] for tree,mask in zip(trees,masking)]))
        weighted_predictions = np.transpose(np.array([ self.weights[j]*tprobs[:,j] for j in range(self.model.n_estimators)]))
        preds=weighted_predictions.sum(axis=1)
        return preds
    
    #Prediction function for multi-class 
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







   
 





