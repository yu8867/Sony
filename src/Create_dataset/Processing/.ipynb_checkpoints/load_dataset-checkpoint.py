import pandas as pd

def Load_dataset():
    tr_path = "C:/Users/yu886/OneDrive/デスクトップ/github/Sony/dataset/train.csv"
    te_path = "C:/Users/yu886/OneDrive/デスクトップ/github/Sony/dataset/test.csv"
    
    train = pd.read_csv(tr_path)
    test = pd.read_csv(te_path)
    
    return train, test