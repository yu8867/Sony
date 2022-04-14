import pandas as pd
import numpy as np

from load_dataset import Load_dataset
from processing_dataset import Processing
from sklearn.model_selection import train_test_split

train, test = Load_dataset()
# 加工・処理
train_df = Processing(train)
test_df = Processing(test)

X = train_df.drop(columns="id")
target = train_df["pm25_mid"]

X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.2, random_state=123)

# index_df, X_train_df, X_test_df, y_train_df, y_test_df = Split(train_df)
# 学習


