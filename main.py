import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from utils import load_housing_data
from transformer import CombineAttributesAdder

imputer = SimpleImputer(strategy='median')
# ordinal_encoder = OrdinalEncoder()
category_encoder = OneHotEncoder()

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
# housing = strat_train_set.copy()

# Data correlations
# housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
# housing['bedrooms_per_room'] = housing['total_bedrooms'] / \
#     housing['total_rooms']
# housing['population_per_household'] = housing['population'] / \
#     housing['households']

# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=.1,
#              s=housing['population'] / 100, label='population',
#              figsize=(10, 7), c='median_house_value',
#              cmap=plt.get_cmap('jet'), colorbar=True)
# plt.legend()

# correlation_matrix = housing.corr()
# print(correlation_matrix['median_house_value'].sort_values(ascending=False))

# attributes = ['median_house_value', 'median_income',
#               'total_rooms', 'housing_median_age']

# scatter_matrix(housing[attributes], figsize=(12, 8))

# housing.plot(kind='scatter', x='median_income',
#              y='median_house_value', alpha=.1)

# test_set.hist(bins=50, figsize=(20, 15))

# plt.show()

# housing['income_cat'].hist()
# plt.show()

"""
    DATA TRANSFORMATION
"""

housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

housing_numeric_values = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_numeric_values)

housing_category_values = housing[['ocean_proximity']]
# housing_category_encoder = ordinal_encoder.fit_transform(
#     houseing_category_values)
housing_category_encoder = category_encoder.fit_transform(
    housing_category_values)

# print(housing_category_encoder.toarray())
X = imputer.transform(housing_numeric_values)

attr_adder = CombineAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# print(housing_extra_attribs)


"""
    PIPELINES
"""

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombineAttributesAdder()),
    ('std_scaler', StandardScaler())
])


housing_num_tr = num_pipeline.fit_transform(housing_numeric_values)

numeric_attributes = list(housing_numeric_values)
category_attributes = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('numeric', num_pipeline, numeric_attributes),
    ('categoric', OneHotEncoder(), category_attributes)
])

housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared)
