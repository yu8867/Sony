import pandas as pd 
from sklearn.model_selection import train_test_split

# ['id', 'year', 'month', 'day', 'Country', 'City', 'lat', 'lon', 'co_cnt',
#        'co_min', 'co_mid', 'co_max', 'co_var', 'o3_cnt', 'o3_min', 'o3_mid',
#        'o3_max', 'o3_var', 'so2_cnt', 'so2_min', 'so2_mid', 'so2_max',
#        'so2_var', 'no2_cnt', 'no2_min', 'no2_mid', 'no2_max', 'no2_var',
#        'temperature_cnt', 'temperature_min', 'temperature_mid',
#        'temperature_max', 'temperature_var', 'humidity_cnt', 'humidity_min',
#        'humidity_mid', 'humidity_max', 'humidity_var', 'pressure_cnt',
#        'pressure_min', 'pressure_mid', 'pressure_max', 'pressure_var',
#        'ws_cnt', 'ws_min', 'ws_mid', 'ws_max', 'ws_var', 'dew_cnt', 'dew_min',
#        'dew_mid', 'dew_max', 'dew_var', 'pm25_mid']

# Processing
def Processing(df):
    drop_cols = ['year', 'month', 'day', 'Country', 'City', 'lat', 'lon']
    df.drop(columns = drop_cols, inplace=True)
    return df

# Split
# def Split(df):
#     col = ["id"]
#     X = df.drop(columns=col)
#     target = df["pm25_mid"]
#     X_train, X_test, y_train, y_test = train_test_split(X,target,test_size=0.2, random_state=123)
#     return id, X_train, X_test, y_train, y_test