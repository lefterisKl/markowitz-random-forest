from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, precision_score
from tree_weighting.MarkowitzForest import MarkowitzForest
from tree_weighting.ScoreWeightedForest import ScoreWeightedForest
from tree_weighting.OOBWeightedForest import OOBWeightedForest
from matplotlib import pyplot as plt
from functools import reduce
from sklearn.ensemble import ExtraTreesClassifier
from tree_weighting.IndependentVarianceForest import IndependentVarianceForest
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,precision_recall_curve
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
from sklearn.metrics import f1_score,precision_recall_curve,precision_score,recall_score
from hyperopt import hp,STATUS_OK,fmin,tpe,Trials
from functools import partial
import tensorflow as tf
import time
import random
from data_loaders import load_dataset,get_classification_datasets
from hyperopt import hp
from hyperopt.pyll.base import scope
from copy import deepcopy
from sklearn.metrics import roc_auc_score,auc
from sklearn.metrics import precision_recall_curve,average_precision_score



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
        'gini', 'entropy'
    ]),
     'class_weight': hp.choice('class_weight', [
        None,  
        'balanced'
    ])

}


xgb_search_space = {
    'objective': hp.choice('objective', ['binary:logistic']),  # Objective function 
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



from sklearn.utils.class_weight import compute_class_weight
def sample_weights_xgb(y_train):
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(zip(np.unique(y_train), weights))
    sample_weights = np.array([class_weights[k] for k in y_train])
    return sample_weights




def prauc(real,probs):
    precision, recall, thresholds = precision_recall_curve(real,probs)
    auc_precision_recall = auc(recall, precision)   
    return auc_precision_recall




def try_model_cv(hyperparams,X,y,TreeEnsemble=RandomForestClassifier,WeightedClassifier=None,rf_params=None,rs=0,mrf_fixed=None):
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
        skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=rs)
        val_losses = []
        train_losses = []
        # Loop over each fold
        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
            Xf_train, Xf_val = X.iloc[train_index], X.iloc[val_index]
            yf_train, yf_val = y.iloc[train_index], y.iloc[val_index]
            if rf_params is not None:
                rf = TreeEnsemble(n_jobs=-1,random_state=rs,**rf_params)
            else:
                rf = TreeEnsemble(n_jobs=-1,random_state=rs,**params)
            if TreeEnsemble == XGBClassifier:
                _=rf.fit(Xf_train.values,yf_train.ravel(),sample_weight = sample_weights_xgb(yf_train))
            else:
                _=rf.fit(Xf_train.values,yf_train.ravel())
            if WeightedClassifier is None:
                prediction_train = rf.predict(Xf_train.values)
                prediction_val = rf.predict(Xf_val.values)
            else:
                if WeightedClassifier == MarkowitzForest:
                    weighted_clf = WeightedClassifier(rate=rate,max_rate=max_rate,gamma=gamma,risk_aversion=risk_aversion)
                else:
                    weighted_clf = WeightedClassifier()
                weighted_clf.set_base_model(rf)
                _=weighted_clf.estimate_weights_clf(Xf_train.values,yf_train.values,Xf_val.values,yf_val.values,oob_estimate=True)
                masking =  [int(w>=0) for w in weighted_clf.weights]
                weighted_clf.weights = np.abs(weighted_clf.weights).tolist()
                weighted_clf.weights = [x/np.sum(weighted_clf.weights) for x in weighted_clf.weights]
                weighted_clf.masking = masking
                prediction_train = (weighted_clf.weighted_predict_proba_n(Xf_train.values) > 0.5).astype(int)
                prediction_val = (weighted_clf.weighted_predict_proba_n(Xf_val.values) > 0.5).astype(int)
            val_losses.append(-f1_score(yf_val,prediction_val))
            train_losses.append(-f1_score(yf_train,prediction_train))
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





names = get_classification_datasets()

for dataset_name in names:
    X,y = load_dataset(dataset_name)
    (y==1).sum() / y.shape[0]
    errors=[]
    elapsed = None
    all_results=[]
    precomputed_rf = dict({})
    from xgboost import XGBClassifier
    mrf_fixed = {'gamma': 0.1, 'max_rate': None, 'rate':0, 'risk_aversion': 0.5}
    best_params = dict({})
    curves=dict([])
    weights_all = dict([])
    for rs in range(0,20):
        X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=rs,stratify=y)
        weighted_methods = [
                            (XGBClassifier, None,xgb_search_space ),
                            (RandomForestClassifier,None,rf_search_space),
                            (ExtraTreesClassifier,None,rf_search_space),
                            (RandomForestClassifier,IndependentVarianceForest,"rf"),
                            (RandomForestClassifier,ScoreWeightedForest,"rf"),
                            (RandomForestClassifier,OOBWeightedForest,"rf"),
                            (RandomForestClassifier,MarkowitzForest,"rf")        ]
        performance = dict({"method":[],'mae':[],"rmse":[],'mape':[],'r2':[]})
        performance = dict({"method":[],'accuracy':[],"f1_score":[],"prauc":[],'roc':[],'aps':[], "precision":[],"recall":[],'elapsed':[]})
        methods = ['xgb','rf',"xt","ivf_rf", "swf_rf","owf_rf",'mrf_rf']
        search_only_rf = False
        i=0
        for TreeEnsemble, WeightedClassifier,search_space in weighted_methods:
            if search_space not in ["rf",'xt']:
                try_model_baked= partial(try_model_cv,X=X_train,y = y_train,TreeEnsemble=TreeEnsemble,WeightedClassifier=WeightedClassifier,rf_params=None,rs=rs,mrf_fixed=mrf_fixed )
                trials = Trials()
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
            if TreeEnsemble == XGBClassifier:
                start = time.time()
                xgb = TreeEnsemble(n_jobs=-1,random_state=rs,**best_params[(methods[i],rs)])
                _=xgb.fit(X_train.values,y_train.ravel())
                end = time.time()
                elapsed = (end - start)
                predict_test = xgb.predict(X_test)
                predict_test_prob = xgb.predict_proba(X_test)[:,1]
            elif WeightedClassifier is None:
                weights_lst = []
                all_trees = []
                skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=rs)
                start = time.time()
                for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
                    Xf_train, Xf_val = X_train.iloc[train_index], X_train.iloc[val_index]
                    yf_train, yf_val = y_train.iloc[train_index], y_train.iloc[val_index]
                    rf = TreeEnsemble(n_jobs=-1,random_state=rs,**best_params[(methods[i],rs)])
                    _=rf.fit(Xf_train.values,yf_train.ravel())
                    all_trees.append(rf)
                final_rf = reduce(combine_rfs,all_trees)
                end = time.time()
                elapsed = end - start
                predict_test = final_rf.predict(X_test)
                predict_test_prob = final_rf.predict_proba(X_test)[:,1]
            else:
                if WeightedClassifier==MarkowitzForest or WeightedClassifier==IndependentVarianceForest:
                    weights_lst = []
                    models = []
                    skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=rs)
                    start = time.time()
                    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
                        Xf_train, Xf_val = X_train.iloc[train_index], X_train.iloc[val_index]
                        yf_train, yf_val = y_train.iloc[train_index], y_train.iloc[val_index]
                        rf = TreeEnsemble(n_jobs=-1,random_state=rs,**best_params[(methods[i],rs)])
                        _=rf.fit(Xf_train.values,yf_train.ravel())
                        weighted_clf = WeightedClassifier(**mrf_fixed)
                        weighted_clf.set_base_model(rf)
                        _=weighted_clf.estimate_weights_clf(Xf_train.values,yf_train.values,Xf_val.values,yf_val.values,oob_estimate=True,rates=True)
                        models.append(weighted_clf)
                    all_rfs = [m.model for m in models]
                    final_rf = reduce(combine_rfs,all_rfs)
                    all_weights = [m.weights for m in models]
                    all_weights = reduce(lambda x,y :x+y, all_weights)
                    masking =  [(np.array(m.weights)>=0).astype(int).tolist() for m in models]
                    masking = reduce(lambda x,y :x+y, masking)
                    weighted_clf = WeightedClassifier(**mrf_fixed)
                    weighted_clf.set_base_model(final_rf)
                    wrf_weights = [(np.abs(np.array(m.weights))/(3*np.abs(np.sum(np.array(m.weights))))).tolist() for m in models]
                    wrf_weights = reduce(lambda x,y :x+y, wrf_weights)
                    weighted_clf.weights = [x/np.sum(wrf_weights) for x in wrf_weights]
                    weighted_clf.masking = masking
                    end = time.time()
                    elapsed = (end - start)
                    predict_test = (weighted_clf.weighted_predict_proba_n(X_test.values) > 0.5).astype(int)
                    predict_test_prob = weighted_clf.weighted_predict_proba_n(X_test.values) 
                    weights_all[(methods[i],rs)] = weighted_clf.weights
                else:
                    weights_lst = []
                    models = []
                    skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=rs)
                    start = time.time()
                    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
                        Xf_train, Xf_val = X_train.iloc[train_index], X_train.iloc[val_index]
                        yf_train, yf_val = y_train.iloc[train_index], y_train.iloc[val_index]
                        rf = TreeEnsemble(n_jobs=-1,random_state=rs,**best_params[(methods[i],rs)])
                        _=rf.fit(Xf_train.values,yf_train.ravel())
                        weighted_clf = WeightedClassifier()
                        weighted_clf.set_base_model(rf)
                        _=weighted_clf.estimate_weights_clf(X_train.values,y_train.values)
                        models.append(weighted_clf)
                    all_weights = [m.weights.tolist() for m in models]
                    all_weights = reduce(lambda x,y :x+y, all_weights)
                    all_rfs = [m.model for m in models]
                    final_rf = reduce(combine_rfs,all_rfs)
                    weighted_clf = WeightedClassifier()
                    weighted_clf.set_base_model(final_rf)
                    weighted_clf.weights = [x/np.sum(all_weights) for x in all_weights]
                    end = time.time()
                    elapsed = (end - start)
                    predict_test = (weighted_clf.weighted_predict_proba(X_test.values) > 0.5).astype(int)
                    predict_test_prob = weighted_clf.weighted_predict_proba(X_test.values)
                    weights_all[(methods[i],rs)] = weighted_clf.weights
            prt = best_params[(methods[i],rs)]
            current_prauc = prauc(y_test,predict_test_prob)
            current_roc = roc_auc_score(y_test,predict_test_prob)
            current_aps = average_precision_score(y_test,predict_test_prob)
            PR,RE,THR = precision_recall_curve(y_test,predict_test_prob)
            curves[methods[i]] =  curves.get( methods[i],[]) + [(PR,RE,THR)]
            accs = accuracy_score(y_test,predict_test)
            f1s = f1_score(y_test,predict_test)
            precision = precision_score(y_test,predict_test)
            recall = recall_score(y_test,predict_test)
            print(f"{methods[i]} {rs}: {f1s} {current_prauc} {precision} {recall}")
            print(prt)
            performance['method'].append(methods[i]) #
            performance['accuracy'].append(accs)
            performance['f1_score'].append(f1s)
            performance['prauc'].append(current_prauc)
            performance['roc'].append(current_roc)
            performance['aps'].append(current_aps)
            performance['precision'].append(precision)
            performance['recall'].append(recall)
            performance['elapsed'].append(elapsed)
            i=i+1
        perf=pd.DataFrame(performance)
        print(perf)
        all_results.append( perf)
    #
    tt = pd.concat(all_results).drop_duplicates()
    tt.to_csv(f"results_binary_classification/{dataset_name}_run.csv",index=False)
    curve_df_all=[]
    for method in curves.keys():
        method_np = [np.transpose(np.array([[i]*len(x[0]),x[0],x[1]])) for i,x in enumerate(curves[method])]
        method_curve = pd.DataFrame(np.concatenate(method_np,axis=0),columns=['run','precision','recall'])
        method_curve['method'] = method
        method_curve['dataset'] = dataset_name
        curve_df_all.append(method_curve)
    curve_df = pd.concat(curve_df_all,axis=0)
    pd.DataFrame(curve_df).to_csv(f"results_binary_classification/{dataset_name}_curves.csv",index=False)





