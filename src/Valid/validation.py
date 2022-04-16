import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def rmse(y_pred, y_true):
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    print("RMSE: ",rmse)
    
def mse(y_pred, y_true):
    mse = mean_squared_error(y_pred, y_true)
    print("MSE: ",mse)
    
def r_2(y_pred, y_true):
    r2 = r2_score(y_pred, y_true)
    print("R^2: ",r2)

# 多くの外れ値が存在するデータの誤差を評価したい
def mae(y_pred, y_true):
    mae = mean_absolute_error(y_pred, y_true)
    print("MAE: ",mae)
    
