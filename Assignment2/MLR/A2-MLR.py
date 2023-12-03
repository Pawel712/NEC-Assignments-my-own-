# This code creates a linear regression model for dataset bank and heart.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#dataset = pd.read_csv('../ModifiedDatasets/bank-additionalModified.csv', sep=';')
dataset = pd.read_csv('../ModifiedDatasets/HeartAttackModified.csv', sep=';')

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

# ------------------- Classification error -------------

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

n_00, n_01, n_10, n_11 = 0, 0, 0, 0

for true_label, predicted_label in zip(y_test, y_pred_binary):
    if true_label == 0 and predicted_label == 0:
        n_00 += 1
    elif true_label == 0 and predicted_label == 1:
        n_01 += 1
    elif true_label == 1 and predicted_label == 0:
        n_10 += 1
    elif true_label == 1 and predicted_label == 1:
        n_11 += 1

classification_error = 100 * ((n_01 + n_10) / (n_00 + n_11 + n_01 + n_10))
print("Classification Error (%):", classification_error)

# ------------------- Classification error ends -------------s

#------Cross validation ------
cv_predictions = cross_val_predict(model, X_train, y_train, cv=5)  # You can adjust the number of folds (cv) as needed
CrossValidationmae = mean_absolute_error(y_train, cv_predictions)

print("Cross validation error: ", CrossValidationmae)
