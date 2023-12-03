import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold


ringTrainData = pd.read_csv('../A2-ring/A2-ring-separable.txt', sep='\t', header=None)
ringTestData = pd.read_csv('../A2-ring/A2-ring-test.txt', sep='\t', header=None)

X_train = ringTrainData.iloc[:,:-1].values
y_train = ringTrainData.iloc[:, -1].values
X_test = ringTestData.iloc[:,:-1].values
y_test = ringTestData.iloc[:,-1].values

X = ringTrainData.iloc[:,:-1].values
y = ringTrainData.iloc[:,-1].values

learning_rate = 0.1
momentum = 0.9
activation_function = 'relu'  
epochs = 15
loss_function = 'binary_crossentropy'

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=activation_function, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation=activation_function),
    tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
])

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)

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
print(f"Error from cross validation: {average_test_loss:.4f}")

# Error from the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Classfication error from test set: {test_loss}")