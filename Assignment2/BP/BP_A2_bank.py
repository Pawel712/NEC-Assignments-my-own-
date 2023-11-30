import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

bankData = pd.read_csv('../ModifiedDatasets/bank-additionalModified.csv', sep=';')

X = bankData.iloc[:,:-1].values
y = bankData.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Parameters for NN
learning_rate = 0.2 # step size of each iteration, how quicly the model learns
momentum = 0.5
activation_function = 'relu'  
epochs = 5
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
print(f"Test Loss: {test_loss}")
#predictions = model.predict(X_test)

#np.savetxt("A2-ring-predictions.csv", predictions, delimiter=",")