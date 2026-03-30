import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# load the dataset/ csv file
california_house_prices = pd.read_csv("C:/Codefile/housing file/housing.csv")

# --- Preprocessing data ---
# fill in the missing values (the missing values are usually present in total_bedrooms)
california_house_prices["total_bedrooms"] = california_house_prices["total_bedrooms"].fillna(california_house_prices["total_bedrooms"].median())

# mapping (converting categorical to numerical)
mapping = {
    "INLAND": 0,
    "NEAR OCEAN": 1,
    "NEAR BAY": 2,
    "ISLAND": 3,
    "<1H OCEAN": 4
}

california_house_prices["ocean_proximity"] = california_house_prices["ocean_proximity"].map(mapping)
