# github repo link for my own reference: https://github.com/rrri01/DATA-221-Group-Project-Raines-Stuff
# group repository link for my own reference: https://github.com/chloeptrsn/DATA-221-Project.git

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import numpy as np

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

housing_data = pd.read_csv('housing.csv', delimiter=',')

# the next few lines were just to figure out what the categorical options were
# d = {}
# l = []
# for i in housing_data["ocean_proximity"]:
#     l.append(i)
#
# for x in l:
#     d[x] = 0
# print(d)

# target: housing_data["median_house_value"]
# categorical: ocean_proximity
# CATEGORICAL OPTIONS:
# NEAR BAY
# <1H OCEAN
# INLAND
# NEAR OCEAN
# ISLAND


housing_data = housing_data.replace({"NEAR BAY": 0, "<1H OCEAN": 1, "INLAND": 2, "NEAR OCEAN": 3, "ISLAND": 4})


# creates feature matrix X of all columns except "median_house_value" and create label vector y as "median_house_value"
X = housing_data.drop("median_house_value", axis=1)
y = housing_data["median_house_value"]

# splits the data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

