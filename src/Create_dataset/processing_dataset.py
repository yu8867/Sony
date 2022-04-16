import pandas as pd 
from sklearn.model_selection import train_test_split

# Processing
def Processing(df):
    drop_cols = ['year', 'month', 'day', 'Country', 'City', 'lat', 'lon']
    df.drop(columns = drop_cols, inplace=True)
    return df

# Split
def Split(df):
    col = ["id"]
    index = df[col]
    X = df.drop(columns=col)
    target = df["pm25_mid"]
    X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.2, random_state=123)
    return X_train, X_test, y_train, y_test, index

def Valid(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2, random_state=123)
    return X_train, X_valid, y_train, y_valid

