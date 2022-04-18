import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Processing
def engin(df):
    drop_cols = ['year', 'month', 'day', 'Country', 'City', 'lat', 'lon']
    
    df["co_std"] = np.sqrt(df["co_var"])
    df["o3_std"] = np.sqrt(df["o3_var"])
    df["so2_std"] = np.sqrt(df["so2_var"])
    df["no2_std"] = np.sqrt(df["no2_var"])
    df["temperature_std"] = np.sqrt(df["temperature_var"])
    df["humidity_std"] = np.sqrt(df["humidity_var"])
    df["pressure_std"] = np.sqrt(df["pressure_var"])
    df["ws_std"] = np.sqrt(df["ws_var"])
    df["dew_std"] = np.sqrt(df["dew_var"])
    
    df.drop(columns = drop_cols, inplace=True)
    return df