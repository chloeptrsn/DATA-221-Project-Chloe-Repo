import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

california_house_prices = pd.read_csv("housing.csv")

# find the median
california_house_prices["total_bedrooms"] = (california_house_prices["total_bedrooms"].
                                             fillna(california_house_prices["total_bedrooms"].median()))

feature_matrix = california_house_prices.loc[:, ['number of rooms', 'house size']]
target_prices = california_house_prices.loc[:, ['price']]

features_train, features_test, prices_train, prices_test = train_test_split(
    feature_matrix,
    target_prices,
    test_size=0.33,
    random_state=42
)