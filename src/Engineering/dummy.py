import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def engin(df):
    drop_cols = ['year', 'month', 'day', 'Country', 'City', 'lat',
                'lon','co_var', 'o3_var','so2_var', 'no2_var',
                'temperature_var', 'humidity_var', 'pressure_var',
                'ws_var','dew_var']
    
    # month
    # month = pd.get_dummies(df['month'], columns=["January","February","March", "April","May",
    # "June","July","August","September","October","November","December"])
    month = pd.get_dummies(df['month'], prefix="month")
    df = pd.concat([df, month], axis=1)
    
    # year
    year = pd.get_dummies(df['year'], prefix="year")
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
    
    df = df.drop(drop_cols ,axis=1)
    return df

def Country(country):
    oseania = ["Australia"]
    asia = ["China", 'India', 'Iran', 'Israel', 'Japan','South Korea', 'Taiwan', 'Thailand', "Vietnam", 'Russia']
    europe = ['Belgium','France', 'Germany','Hungary','Italy','Poland','Serbia','Spain','Turkey',
              'United Kingdom','Netherlands','Bosnia and Herzegovina','Croatia']
    south_america = ['Brazil','Chile']
    north_america = ['Canada','Mexico','United States']
    africa = ['South Africa']
    
    if country in oseania:
        return "oseania"
    elif country in asia:
        return "asia"
    elif country in europe:
        return "europe"
    elif country in south_america:
        return "south_america"
    elif country in north_america:
        return "north_america"
    else:
        return "africa"

def engin_continet(df):
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
    
    Continet = df["Country"].apply(Country)
    Continet = pd.get_dummies(Continet)
    df = pd.concat([df, Continet], axis=1)
    
    df["co_std"] = np.sqrt(df["co_var"])
    df["o3_std"] = np.sqrt(df["o3_var"])
    df["so2_std"] = np.sqrt(df["so2_var"])
    df["no2_std"] = np.sqrt(df["no2_var"])
    df["temperature_std"] = np.sqrt(df["temperature_var"])
    df["humidity_std"] = np.sqrt(df["humidity_var"])
    df["pressure_std"] = np.sqrt(df["pressure_var"])
    df["ws_std"] = np.sqrt(df["ws_var"])
    df["dew_std"] = np.sqrt(df["dew_var"])
    
    df = df.drop(drop_cols ,axis=1)
    return df