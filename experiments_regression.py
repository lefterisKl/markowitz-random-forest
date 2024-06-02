from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, precision_score
from tree_weighting.MarkowitzForest import MarkowitzForest
from tree_weighting.IndependentVarianceForest import IndependentVarianceForest
from tree_weighting.ScoreWeightedForest import ScoreWeightedForest
from tree_weighting.OOBWeightedForest import OOBWeightedForest
from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
from hyperopt import hp,STATUS_OK,fmin,tpe,Trials
from functools import partial
import tensorflow as tf
import time
import random
from data_loaders import load_dataset
from hyperopt import hp
from hyperopt.pyll.base import scope
from functools import reduce
from sklearn.ensemble import ExtraTreesRegressor
import time
from data_loaders import get_regression_datasets
from copy import deepcopy
from xgboost import XGBRegressor
from hyperopt import hp,STATUS_OK,fmin,tpe,Trials
from sklearn.model_selection import KFold


# Define the search space for max_depth, min_samples_split, and min_samples_leaf
rf_search_space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 200, 10)),
    'max_depth': hp.choice('max_depth', [None,scope.int(hp.quniform('max_depth_int', 2, 10, 1))]),
    'min_samples_split': hp.choice('min_samples_split', [
        scope.int(hp.quniform('min_samples_split_int', 2, 20, 1)),  
        hp.uniform('min_samples_split_frac', 0.01, 0.5)  
    ]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [
        scope.int(hp.quniform('min_samples_leaf_int', 1, 20, 1)), 
        hp.uniform('min_samples_leaf_frac', 0.01, 0.2)  
    ]),
    'max_features': hp.choice('max_features', ['sqrt', 'log2'
        ]),
    'criterion': hp.choice('criterion', [
        'squared_error'
    ]),
}



xgb_search_space = {
    'objective': hp.choice('objective', ['reg:squarederror']),  # Objective function for ranking
    'booster': hp.choice('booster', ['gbtree']),  # Booster type
    'learning_rate': hp.loguniform('learning_rate', -5, 0),  # Log scale for learning rate
    'gamma': hp.uniform('gamma', 0, 5),  # Minimum loss reduction required to make a further partition on a leaf node
    'max_depth': scope.int(hp.quniform('max_depth', 1, 7, 1)),  # Depth of trees
    'min_child_weight': hp.uniform('min_child_weight', 1, 10),  # Minimum sum of instance weight (hessian) needed in a child
    'max_delta_step': scope.int(hp.quniform('max_delta_step', 0, 10, 1)),  # Maximum delta step
    'n_estimators': scope.int(hp.quniform('n_estimators', 5, 105, 5)),  # Number of estimators
    'subsample': hp.uniform('subsample', 0.5, 1),  # Subsample ratio of the training instances
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),  # Subsample ratio of columns when constructing each tree
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1),  # Subsample ratio of columns for each level
    'colsample_bynode': hp.uniform('colsample_bynode', 0.3, 1),  # Subsample ratio of columns for each split
    'lambda': hp.uniform('lambda', 1, 5),  # L2 regularization term on weights
    'alpha': hp.uniform('alpha', 0, 1),  # L1 regularization term on weights
    'grow_policy': hp.choice('grow_policy', ["depthwise", "lossguide"]),  # Grow policy
    'max_bin': scope.int(hp.quniform('max_bin', 64, 768, 64))  # Maximum bin
}




def try_model_cv(hyperparams,X,y,TreeEnsemble=RandomForestRegressor,WeightedRegressor=None,rf_params=None,rs=0,mrf_fixed=None):
    try:
        params = deepcopy(hyperparams)
        if mrf_fixed is not None:
            params.update(mrf_fixed)
        if 'rate' in params:
            rate = params['rate']
            max_rate = params['max_rate']
            gamma = params['gamma']
            risk_aversion = params['risk_aversion']
            del params['rate']
            del params['max_rate']
            del params['gamma']
            del params['risk_aversion']
        # Setup stratified 3-fold cross-validation
        skf = KFold(n_splits=3,shuffle=True,random_state=rs)
        val_losses = []
        train_losses = []
        # Loop over each fold
        for fold, (train_index, val_index) in enumerate(skf.split(X)):
            Xf_train, Xf_val = X.iloc[train_index], X.iloc[val_index]
            yf_train, yf_val = y.iloc[train_index], y.iloc[val_index]
            if rf_params is not None:
                rf = TreeEnsemble(n_jobs=-1,random_state=rs,**rf_params)
            else:
                rf = TreeEnsemble(n_jobs=-1,random_state=rs,**params)
            _=rf.fit(Xf_train.values,yf_train.ravel())
            if WeightedRegressor is None:
                prediction_train = rf.predict(Xf_train.values)
                prediction_val = rf.predict(Xf_val.values)
            else:
                if WeightedRegressor == MarkowitzForest:
                    weighted_reg = WeightedRegressor(rate=rate,max_rate=max_rate,gamma=gamma,risk_aversion=risk_aversion)
                else:
                    weighted_reg = WeightedRegressor()
                weighted_reg.set_base_model(rf)
                _=weighted_reg.estimate_weights_reg(Xf_train.values,yf_train.values,Xf_val.values,yf_val.values,oob_estimate=True)
                masking =  [int(w>=0) for w in weighted_reg.weights]
                weighted_reg.weights = np.abs(weighted_reg.weights).tolist()
                weighted_reg.weights = [x/np.sum(weighted_reg.weights) for x in weighted_reg.weights]
                weighted_reg.masking = masking
                prediction_train = weighted_reg.weighted_predict(Xf_train.values) 
                prediction_val = weighted_reg.weighted_predict(Xf_val.values) 
            val_losses.append(mean_squared_error(yf_val,prediction_val))
            train_losses.append(mean_squared_error(yf_train,prediction_train))
        val_loss = np.mean(val_losses)
        train_loss = np.mean(train_losses)
        result={'loss': val_loss,'train_loss':train_loss, 'status': STATUS_OK}
    except Exception as e:
        print(e)
        result = {'loss': np.inf,'train_loss':np.inf, 'status': STATUS_FAIL}
    return result


def combine_rfs(rf_a, rf_b):
    rf_a_clone = deepcopy(rf_a)
    rf_a_clone.estimators_ += rf_b.estimators_
    rf_a_clone.n_estimators = len(rf_a_clone.estimators_)
    return rf_a_clone


names = get_regression_datasets()
for dataset_name in names:
    X,y = load_dataset(dataset_name)
    (y==1).sum() / y.shape[0]
    errors=[]
    rs = 0
    # https://datascience.stackexchange.com/a/93680
    all_scores = []
    all_params = {"rf":[],"mf":[],"owf":[],"swf":[]}
    all_times = {"rf":[],"mf":[],"owf":[],"swf":[]}
    all_weights = {"rf":[],"mf":[],"owf":[],"swf":[]}
    rs=0
    elapsed = None
    all_results=[]
    precomputed_rf = dict({})
    mrf_fixed = {'gamma': 0.1, 'max_rate': None, 'rate':0, 'risk_aversion': 0.5}
    best_params = dict({})
    curves=dict([])
    for rs in range(0,20):
        X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=rs)
        weighted_methods = [
                            (XGBRegressor, None,xgb_search_space ),
                            (RandomForestRegressor,None,rf_search_space),
                            (ExtraTreesRegressor,None,rf_search_space),
                            (RandomForestRegressor,IndependentVarianceForest,"rf"),
                            (RandomForestRegressor,ScoreWeightedForest,"rf"),
                            (RandomForestRegressor,OOBWeightedForest,"rf"),
                            (RandomForestRegressor,MarkowitzForest,"rf")        ]
        performance = dict({"method":[],'mae':[],"rmse":[],'mape':[],'r2':[],'time':[]})
        methods = ['xgb','rf',"xt","ivf_rf", "swf_rf","owf_rf",'mrf_rf']
        search_only_rf = False
        i=0
        for TreeEnsemble, WeightedRegressor,search_space in weighted_methods:
            if search_space not in ["rf",'xt']:
                try_model_baked= partial(try_model_cv,X=X_train,y = y_train,TreeEnsemble=TreeEnsemble,WeightedRegressor=WeightedRegressor,rf_params=None,rs=rs,mrf_fixed=mrf_fixed )
                trials = Trials()
                #if rs not in precomputed_rf:
                random.seed(0)
                np.random.seed(0)
                tf.random.set_seed(0)
                rstate = np.random.default_rng(42)
                start_time = time.time()
                best = fmin(
                    fn=try_model_baked,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=20,
                    trials=trials,
                    rstate=rstate
                )
                duration =  time.time() - start_time
                best_score = -min([trial['result']['loss'] for trial in trials.trials])
                best_space = hyperopt.space_eval(search_space,best)
                results = pd.Series(np.array([-trial['result']['loss'] for trial in trials.trials])).values.tolist()
                metrics=trials.trials[results.index(best_score)]['result']
                training_results = pd.Series(np.array([-trial['result']['train_loss'] for trial in trials.trials]))
                best_params[(methods[i],rs)] = best_space
                best_space_rf = {key:value for key,value in best_space.items() if key in rf_search_space.keys()}
                precomputed_rf[rs] = best_space_rf
            else:
                if "_rf" in methods[i]:
                    best_params[(methods[i],rs)] = best_params[("rf",rs)]
                elif "_xt" in methods[i]:
                    best_params[(methods[i],rs)] = best_params[("xt",rs)]
            if TreeEnsemble == XGBRegressor:
                start = time.time()
                xgb = TreeEnsemble(n_jobs=-1,random_state=rs,**best_params[(methods[i],rs)])
                _=xgb.fit(X_train.values,y_train.ravel())
                end = time.time()
                elapsed = end-start
                predict_test = xgb.predict(X_test)
            elif WeightedRegressor is None:
                weights_lst = []
                all_trees = []
                skf = KFold(n_splits=3,shuffle=True,random_state=rs)
                start = time.time()
                for fold, (train_index, val_index) in enumerate(skf.split(X_train)):
                    Xf_train, Xf_val = X_train.iloc[train_index], X_train.iloc[val_index]
                    yf_train, yf_val = y_train.iloc[train_index], y_train.iloc[val_index]
                    rf = TreeEnsemble(n_jobs=-1,random_state=rs,**best_params[(methods[i],rs)])
                    _=rf.fit(Xf_train.values,yf_train.ravel())
                    all_trees.append(rf)
                final_rf = reduce(combine_rfs,all_trees)
                start = time.time()
                end = time.time()
                elapsed = end-start
                predict_test = final_rf.predict(X_test)
            else:
                if WeightedRegressor==MarkowitzForest or WeightedRegressor==IndependentVarianceForest:
                    weights_lst = []
                    models = []
                    skf = KFold(n_splits=3,shuffle=True,random_state=rs)
                    start = time.time()
                    for fold, (train_index, val_index) in enumerate(skf.split(X_train)):
                        Xf_train, Xf_val = X_train.iloc[train_index], X_train.iloc[val_index]
                        yf_train, yf_val = y_train.iloc[train_index], y_train.iloc[val_index]
                        rf = TreeEnsemble(n_jobs=-1,random_state=rs,**best_params[(methods[i],rs)])
                        _=rf.fit(Xf_train.values,yf_train.ravel())
                        weighted_reg = WeightedRegressor(**mrf_fixed)
                        weighted_reg.set_base_model(rf)
                        _=weighted_reg.estimate_weights_reg(Xf_train.values,yf_train.values,Xf_val.values,yf_val.values,oob_estimate=True,rates=True)
                        models.append(weighted_reg)
                    all_rfs = [m.model for m in models]
                    final_rf = reduce(combine_rfs,all_rfs)
                    all_weights = [m.weights for m in models]
                    all_weights = reduce(lambda x,y :x+y, all_weights)
                    masking =  [(np.array(m.weights)>=0).astype(int).tolist() for m in models]
                    masking = reduce(lambda x,y :x+y, masking)
                    weighted_reg = WeightedRegressor(**mrf_fixed)
                    weighted_reg.set_base_model(final_rf)
                    wrf_weights = [(np.abs(np.array(m.weights))/(3*np.abs(np.sum(np.array(m.weights))))).tolist() for m in models]
                    wrf_weights = reduce(lambda x,y :x+y, wrf_weights)
                    weighted_reg.weights = [x/np.sum(wrf_weights) for x in wrf_weights]
                    weighted_reg.masking = masking
                    end = time.time()
                    elapsed = end-start
                    predict_test = weighted_reg.weighted_predict(X_test.values)
                else:
                    weights_lst = []
                    models = []
                    skf = KFold(n_splits=3,shuffle=True,random_state=rs)
                    start = time.time()
                    for fold, (train_index, val_index) in enumerate(skf.split(X_train)):
                        Xf_train, Xf_val = X_train.iloc[train_index], X_train.iloc[val_index]
                        yf_train, yf_val = y_train.iloc[train_index], y_train.iloc[val_index]
                        rf = TreeEnsemble(n_jobs=-1,random_state=rs,**best_params[(methods[i],rs)])
                        _=rf.fit(Xf_train.values,yf_train.ravel())
                        weighted_reg = WeightedRegressor()
                        weighted_reg.set_base_model(rf)
                        _=weighted_reg.estimate_weights_reg(X_train.values,y_train.values)
                        models.append(weighted_reg)
                    all_weights = [m.weights.tolist() for m in models]
                    all_weights = reduce(lambda x,y :x+y, all_weights)
                    all_rfs = [m.model for m in models]
                    final_rf = reduce(combine_rfs,all_rfs)
                    weighted_reg = WeightedRegressor()
                    weighted_reg.set_base_model(final_rf)
                    weighted_reg.weights = [x/np.sum(all_weights) for x in all_weights]
                    end = time.time()
                    elapsed = end-start
                    predict_test = weighted_reg.weighted_predict(X_test.values) 
            prt = best_params[(methods[i],rs)]
            current_mae = mean_absolute_error(y_test,predict_test)
            current_mse = mean_squared_error(y_test,predict_test)
            current_rmse = np.sqrt(current_mse)
            current_mape = mean_absolute_percentage_error(y_test,predict_test)
            current_r2 = r2_score(y_test,predict_test)
            print(f"{methods[i]} {rs}: {current_mae} {current_rmse} {current_mape} {current_r2}")
            print(prt)
            performance['method'].append(methods[i]) #
            performance['mae'].append(current_mae)
            performance['rmse'].append(current_rmse)
            performance['mape'].append(current_mape)
            performance['r2'].append(current_r2)
            performance['time'].append(elapsed)
            i=i+1
        perf=pd.DataFrame(performance)
        print(perf)
        print(perf)
        all_results.append( perf)
    #
    tt = pd.concat(all_results).drop_duplicates()
    tt.to_csv(f"results_regression/{dataset_name}_run.csv",index=False)