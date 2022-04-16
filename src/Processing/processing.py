import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split
def Split(df):
    index = df["id"]
    target = df["pm25_mid"]
    
    col = ["id","pm25_mid"]
    X = df.drop(columns=col)
    X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.2, random_state=123)
    return X_train, X_test, y_train, y_test, index


def Valid(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2, random_state=123)
    return X_train, X_valid, y_train, y_valid


def Submission(index, predict, name):
    submission = np.stack([index, predict])
    submission = pd.DataFrame(submission).T
    submission.to_csv("submission/submission_{}.csv".format(name),index=False, header=False)
    
def rmse(y_pred, y_true):
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    print(rmse)
    