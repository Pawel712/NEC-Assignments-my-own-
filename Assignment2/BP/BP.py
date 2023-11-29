import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ringTrainData = pd.read_csv('../A2-ring/A2-ring-separable.txt', sep='\t', header=None)
ringTestData = pd.read_csv('../A2-ring/A2-ring-test.txt', sep='\t', header=None)

X_train = ringTrainData.iloc[:,:-1].values
y_train = ringTrainData.iloc[:, -1].values
X_test = ringTestData.iloc[:,:-1].values
y_test = ringTestData.iloc[:,-1].values

# Parameters for NN
learning_rate = 0.01
momentum = 0.9
activation_function = 'relu'  
epochs = 50
loss_function = 'binary_crossentropy'

# Build the neural network, adds layers and amount of neurons in each layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation=activation_function, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation=activation_function),
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
print(f"Test Loss: {test_loss}")
predictions = model.predict(X_test)

#np.savetxt("A2-ring-predictions.csv", predictions, delimiter=",")