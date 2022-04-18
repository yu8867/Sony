import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def engin(df):
    drop_cols = ['year', 'month', 'day', 'Country', 'City', 'lat',
                'lon','co_var', 'o3_var','so2_var', 'no2_var',
                'temperature_var', 'humidity_var', 'pressure_var',
                'ws_var','dew_var']
    
    # month
    month = pd.get_dummies(df['month'])
    df = pd.concat([df, month], axis=1)
    
    # year
    year = pd.get_dummies(df['year'])
    df = pd.concat([df, year], axis=1)
    
    # country
    country = pd.get_dummies(df['Country'])
    df = pd.concat([df, country], axis=1)
    
    df["co_std"] = np.sqrt(df["co_var"])
    df["o3_std"] = np.sqrt(df["o3_var"])
    df["so2_std"] = np.sqrt(df["so2_var"])
    df["no2_std"] = np.sqrt(df["no2_var"])
    df["temperature_std"] = np.sqrt(df["temperature_var"])
    df["humidity_std"] = np.sqrt(df["humidity_var"])
    df["pressure_std"] = np.sqrt(df["pressure_var"])
    df["ws_std"] = np.sqrt(df["ws_var"])
    df["dew_std"] = np.sqrt(df["dew_var"])
    
    df = df.drop(columns = drop_cols)
    return df