import os
import pandas as pd
import numpy as np
from zlib import crc32


def load_housing_data(housing_data_path='datasets'):
    csv_path = os.path.join(housing_data_path, 'housing.csv')
    return pd.read_csv(csv_path)


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def split_train_test(data, test_ratio=0.2):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
