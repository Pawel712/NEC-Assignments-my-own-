import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

bankData = pd.read_csv('../ModifiedDatasets/bank-additionalModified.csv', sep=';')

X = bankData.iloc[:,:-1].values
y = bankData.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Parameters for NN
learning_rate = 0.2
momentum = 0.5
activation_function = 'relu'  
epochs = 5
loss_function = 'mean_squared_error'
    
model = tf.keras.Sequential([
    tf.keras.layers.Dense(200, activation=activation_function, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(200, activation=activation_function, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
])

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")

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

# ------------------- Classification error ends -------------

#-----------------Cross-validation k-fold ---------------------
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=20)

overall_test_acc = 0
overall_test_loss = 0

for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    print(f"Training Fold {fold}")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation=activation_function, input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(200, activation=activation_function),
        tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test)

    overall_test_acc += test_acc
    overall_test_loss += test_loss

# Error from cross validation
average_test_loss = overall_test_loss / num_folds
print(f"Expected classfication error from cross validation: {average_test_loss:.4f}")

# Error from the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Classification error from test set: {test_loss}")

