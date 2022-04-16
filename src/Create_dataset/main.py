import pandas as pd
import numpy as np

from load_dataset import Load_dataset
from processing_dataset import Processing, Split, Valid
from Model.lightgbm import LightGBM

train, test = Load_dataset()

# 加工・処理
train_df = Processing(train)
test_df = Processing(test)

# val, train, test
X_train, X_test, y_train, y_test, index = Split(train_df)
X_train, X_valid, y_train, y_valid = Valid(X_train, y_train)

# 学習
Light_GBM = LightGBM(X_train, X_valid, y_train, y_valid, fig=1)
predict_light_gbm = Light_GBM.predict(y_train)
print(predict_light_gbm)


