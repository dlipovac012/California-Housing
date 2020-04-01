import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

from utils import load_housing_data

housing = load_housing_data()

housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

# Copy housing
housing = strat_train_set.copy()

# Data correlations
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / \
    housing['total_rooms']
housing['population_per_household'] = housing['population'] / \
    housing['households']

# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=.1,
#              s=housing['population'] / 100, label='population',
#              figsize=(10, 7), c='median_house_value',
#              cmap=plt.get_cmap('jet'), colorbar=True)
# plt.legend()

correlation_matrix = housing.corr()
print(correlation_matrix)

attributes = ['median_house_value', 'median_income',
              'total_rooms', 'housing_median_age']

# scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind='scatter', x='median_income',
             y='median_house_value', alpha=.1)

# test_set.hist(bins=50, figsize=(20, 15))

# plt.show()

# housing['income_cat'].hist()
# plt.show()
