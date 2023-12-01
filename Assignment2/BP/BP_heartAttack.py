import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

bankData = pd.read_csv('../ModifiedDatasets/HeartAttackModified.csv', sep=';')

X = bankData.iloc[:,:-1].values
y = bankData.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Parameters for NN
learning_rate = 0.2 # step size of each iteration, how quicly the model learns
momentum = 0.5
activation_function = 'relu'  
epochs = 25
loss_function = 'mean_squared_error'

# Build the neural network, adds layers and amount of neurons in each layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(200, activation=activation_function, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(200, activation=activation_function, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
])

# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")
print(f"Test Error: {test_loss}")


#-----------------Cross-validation k-fold ---------------------
# Specify the number of folds for cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=20)

# Initialize variables to store overall performance metrics
overall_test_acc = 0
overall_test_loss = 0

# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    print(f"Training Fold {fold}")

    # Split data into training and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Build the neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation=activation_function, input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(200, activation=activation_function),
        tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
    ])

    # Compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the test set for this fold
    test_loss, test_acc = model.evaluate(X_test, y_test)

    # Update overall performance metrics
    overall_test_acc += test_acc
    overall_test_loss += test_loss

# Calculate and print average performance metrics across all folds
average_test_acc = overall_test_acc / num_folds
average_test_loss = overall_test_loss / num_folds
print(f"\nAverage Test Accuracy Across Folds: {average_test_acc:.4f}")
print(f"Average Test Error Across Folds: {average_test_loss:.4f}")