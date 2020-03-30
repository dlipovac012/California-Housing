import matplotlib.pyplot as plt

from utils import load_housing_data

housing = load_housing_data()
# print(load_housing_data().describe())

housing.hist(bins=50, figsize=(20, 15))

plt.show()
