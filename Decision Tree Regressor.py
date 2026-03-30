from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd

# Load dataset
housing = pd.read_csv("housing.csv")

# Encoding
housing = pd.get_dummies(housing)
housing = housing.dropna()