

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def get_classification_datasets():
    return ["aids", "bands","bank-marketing","defaults","shoppers"]


def get_regression_datasets():
    return ["car",'insurance',"news","wine","superconduct"] 


def get_multiclass_datasets():
    return ["pen",'plates',"room","statlog image","student"] 

def load_dataset(dataset):
    from ucimlrepo import fetch_ucirepo
    if dataset == "bands":
        data = pd.read_csv("data/bands.data",header=None)
        data.columns = [f"feature_{col}" for col in data.columns[1:]] + ["target"]
        X=data.drop(columns=["target","feature_1","feature_2","feature_3","feature_4"])
        y=(data['target']=="band").astype(int)
        numerics = [ f"feature_{i}" for i in  range(20,40)]
        for feature in numerics:
            try:
                X[feature] = X[feature].replace("?",np.NaN).replace("band",np.NaN).astype(float)
                imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
                X[[feature]] = imp_mean.fit_transform(X[[feature]])
            except:
                print(f"{feature} was not transformed.")     
        X=pd.get_dummies(X)
    elif dataset == "shoppers":
        data = pd.read_csv("data/online_shoppers_intention.csv")
        X=data.drop(columns="Revenue")
        X=pd.get_dummies(X)
        y=(data['Revenue']).astype(int)
    elif dataset == "bank-marketing":
        data = pd.read_csv("data/bank-additional-full.csv",sep=";")
        data.drop_duplicates(inplace=True)
        X=data.drop(columns="y")
        X=pd.get_dummies(X)
        y=(data['y']=="yes").astype(int)
    elif dataset == "defaults":
        data = pd.read_csv("data/default of credit card clients.csv")
        X=data.drop(columns="default payment next month")
        y=(data['default payment next month']).astype(int)
    elif dataset == "news":
        X=pd.read_csv("data/OnlineNewsPopularity.csv")
        X.drop(columns=['url',' timedelta'],inplace=True)
        X.columns = [x.strip() for x in X.columns]
        y = X.shares
        X=X.drop(columns=["shares"])
    elif dataset =="car":
        X=pd.read_csv("data/car_train.csv")
        X.columns = [x.strip() for x in X.columns]
        X.Mileage = X.Mileage.str.replace(" km","").astype("float")
        X.drop(columns=["Model","ID"],inplace=True)
        X["Price"] = X["Price"].astype(float)
        X=pd.get_dummies(X)
        y = X.Price
        X=X.drop(columns=["Price"])
    elif dataset=="wine":
        red = pd.read_csv("data/winequality-red.csv",sep=";")
        white = pd.read_csv("data/winequality-white.csv",sep=";")
        red["color"] = 1
        white["color"]=2
        X = pd.concat([red,white])
        y = X["quality"]
        X = X.drop(columns=["quality"])
    elif dataset=="insurance":
        X=pd.read_csv("data/ticdata2000.txt",sep="\t",header=None)
        X.columns = X.columns.astype(str)
        y=X["85"] 
        X = X.drop(columns=["85"])
    elif dataset=='aids':
        from ucimlrepo import fetch_ucirepo 
        aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890) 
        X = aids_clinical_trials_group_study_175.data.features 
        y = aids_clinical_trials_group_study_175.data.targets 
        return X,y['cid']
    elif dataset=='student':
        x = pd.read_csv("data/student_success.csv",sep=';')
        y = x['Target']
        mapping = dict([(y,x) for x,y in enumerate(y.value_counts().index.tolist())])
        y = y.map(mapping)
        X=x.drop(columns = ['Target'])
    elif dataset=="pen":
        from ucimlrepo import fetch_ucirepo 
        pen_based_recognition_of_handwritten_digits = fetch_ucirepo(id=81) 
        X = pen_based_recognition_of_handwritten_digits.data.features 
        y = pen_based_recognition_of_handwritten_digits.data.targets 
        y=y['Class']
    elif dataset=="plates":
        from ucimlrepo import fetch_ucirepo 
        steel_plates_faults = fetch_ucirepo(id=198) 
        X = steel_plates_faults.data.features 
        y = steel_plates_faults.data.targets 
        y['class'] = y.idxmax(axis=1)
        y = y['class']
        mapping = dict([(y,x) for x,y in enumerate(y.value_counts().index.tolist())])
        y= y.map(mapping) 
    elif dataset == 'statlog image':
        from ucimlrepo import fetch_ucirepo 
        statlog_image_segmentation = fetch_ucirepo(id=147) 
        X = statlog_image_segmentation.data.features 
        y = statlog_image_segmentation.data.targets 
        y=y['class']
        mapping = dict([(y,x) for x,y in enumerate(y.value_counts().index.tolist())])
        y= y.map(mapping)   
    elif dataset == "room":
        from ucimlrepo import fetch_ucirepo 
        room_occupancy_estimation = fetch_ucirepo(id=864)   
        X = room_occupancy_estimation.data.features 
        y = room_occupancy_estimation.data.targets 
        X.drop(columns=['Date','Time'],inplace=True)
        y = y['Room_Occupancy_Count']
    return X,y