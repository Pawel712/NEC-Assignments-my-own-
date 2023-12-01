import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ringTrainData = pd.read_csv('../A2-ring/A2-ring-separable.txt', sep='\t', header=None)
ringTestData = pd.read_csv('../A2-ring/A2-ring-test.txt', sep='\t', header=None)

X_train = ringTrainData.iloc[:,:-1].values
y_train = ringTrainData.iloc[:, -1].values
X_test = ringTestData.iloc[:,:-1].values
y_test = ringTestData.iloc[:,-1].values

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')