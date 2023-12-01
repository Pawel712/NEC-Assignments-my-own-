import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset = pd.read_csv('../ModifiedDatasets/bank-additionalModified.csv', sep=';')
#dataset = pd.read_csv('../ModifiedDatasets/HeartAttackModified.csv', sep=';')
#dataset = pd.read_csv('../ModifiedDatasets/HeartAttackModified.csv', sep=';')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')