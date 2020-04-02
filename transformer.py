import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

rooms_xi, bedrooms_xi, population_xi, households_xi = 3, 4, 5, 6


class CombineAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_xi] / X[:, households_xi]
        population_per_household = X[:, population_xi] / X[:, households_xi]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_xi] / X[:, rooms_xi]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
