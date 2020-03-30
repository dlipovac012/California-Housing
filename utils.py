import os
import pandas as pd

def load_housing_data(housing_data_path='datasets'):
    csv_path = os.path.join(housing_data_path, 'housing.csv')
    return pd.read_csv(csv_path)