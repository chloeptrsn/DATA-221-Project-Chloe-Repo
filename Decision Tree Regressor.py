from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import pandas as pd

# Load dataset
california_house_prices = pd.read_csv("housing.csv")

# Encoding and preprocessing data
california_house_prices = pd.get_dummies(california_house_prices)
california_house_prices = california_house_prices.fillna(california_house_prices.median())

# Split into feature and target variables
feature_matrix = california_house_prices.drop("median_house_value", axis=1)
target_values = california_house_prices["median_house_value"]

# Train test split-- 70% train, 30% test
features_train, features_test, labels_train, labels_test = train_test_split(feature_matrix, target_values, test_size=0.3, random_state=42)

# Creating model
decision_tree = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

# Train model
decision_tree.fit(features_train, labels_train)

# Predict
predicted_labels = decision_tree.predict(features_test)

rmse = root_mean_squared_error(labels_test, predicted_labels)
r2 = r2_score(labels_test, predicted_labels)

print("RMSE:", rmse)
print("R2:", r2)
