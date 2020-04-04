import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error

from data import housing_prepared, housing_labels, housing, full_pipeline


def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation', scores.std())


# train linear regression model
# lin_regression = LinearRegression()
# lin_regression.fit(housing_prepared, housing_labels)
# model = DecisionTreeRegressor()
# model.fit(housing_prepared, housing_labels)

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

model = RandomForestRegressor()

grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring='neg_mean_squared_error',
    return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
model.fit(housing_prepared, housing_labels)


# Testing on some data
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

# print("Predictions:", lin_regression.predict(some_data_prepared))
# print("Labels:", list(some_labels))

housing_predictions = model.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

scores = cross_val_score(model, housing_prepared,
                         housing_labels, scoring='neg_mean_squared_error',
                         cv=10)

model_rmse_scores = np.sqrt(-scores)

# display_scores(model_rmse_scores)
