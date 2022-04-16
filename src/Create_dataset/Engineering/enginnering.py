import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Processing
def engin(df):
    drop_cols = ['year', 'month', 'day', 'Country', 'City', 'lat', 'lon']
    df.drop(columns = drop_cols, inplace=True)
    return df


