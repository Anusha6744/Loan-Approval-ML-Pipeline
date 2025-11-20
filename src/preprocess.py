import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
   return pd.read_csv(path)

def split_features_target(df, target_col):
    x= df.drop(columns=[target_col,"Loan_ID"])
    y = df[target_col]
    return x, y

def train_test_split_data(x, y, test_size=0.2):
     return train_test_split(x, y, test_size=test_size, random_state=42)

