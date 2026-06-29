"""
NOTE: imports are there for every snippet seperately so that testing with collab becomes easier
"""

# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 1.
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 

# Generate data 
x = np.linspace(-10, 10, 100) 
y = sigmoid(x) 

# Plot sigmoid function 
plt.plot(x, y) 
plt.title('Sigmoid Function') 
plt.xlabel('x') 
plt.ylabel('sigmoid(x)') 
plt.grid(True)

# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 2.
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np

def forward_propagation(inputs, weights, biases): 
    # Calculate weighted sum of inputs 
    weighted_sum = np.dot(inputs, weights) + biases 
    
    # Apply sigmoid activation function 
    output = 1 / (1 + np.exp(-weighted_sum)) 
    return output 

# Example inputs 
inputs = np.array([0.5, 0.3, 0.2]) 
weights = np.array([0.4, 0.7, 0.2]) 
biases = 0.1 

# Perform forward propagation 
output = forward_propagation(inputs, weights, biases) 
print("Forward Propagation Output:", output)

# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 3.
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np 

def backpropagation(inputs, weights, biases, targets, learning_rate): 
    # Forward pass 
    weighted_sum = np.dot(inputs, weights) + biases 
    output = 1 / (1 + np.exp(-weighted_sum)) 
    
    # Compute error 
    error = targets - output 
    
    # Compute gradient 
    gradient = error * output * (1 - output) 
    
    # Update weights and biases 
    weights += learning_rate * np.dot(inputs.T, gradient) 
    biases += learning_rate * np.sum(gradient) 
    
    return error 

# Example inputs and targets 
inputs = np.array([[0.5, 0.3, 0.2]]) 
weights = np.array([[0.4], [0.7], [0.2]]) 
biases = 0.1 
targets = np.array([0.8]) 

# Perform backpropagation 
learning_rate = 0.1 
error = backpropagation(inputs, weights, biases, targets, learning_rate) 
print("Backpropagation Error:", error)

# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 4.
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np 
from sklearn.neural_network import MLPClassifier 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

# Load dataset 
iris = load_iris() 
X, y = iris.data, iris.target 

# Split dataset 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model 
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Predictions 
y_pred = clf.predict(X_test)

# Accuracy 
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 5.
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np 
import tensorflow as tf 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 

# Load dataset 
iris = load_iris() 
X, y = iris.data, iris.target 

# Normalize 
scaler = MinMaxScaler() 
X_normalized = scaler.fit_transform(X) 

# Split dataset 
X_train, X_test = train_test_split(X_normalized, test_size=0.2, random_state=42)

# Architecture 
input_dim = X_train.shape[1] 
encoding_dim = 2 

input_layer = tf.keras.layers.Input(shape=(input_dim,)) 
encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer) 
decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder) 

# Model 
autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Train 
autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test)
)

# Encoded output 
encoded_data = autoencoder.predict(X_normalized)

# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 6.
# ---------------------------------------------------------------------------------------------------------------------
import tensorflow as tf 
import numpy as np 

num_states = 4 
num_actions = 2 

class QNetwork(tf.keras.Model): 
    def __init__(self, num_actions): 
        super(QNetwork, self).__init__() 
        self.dense1 = tf.keras.layers.Dense(32, activation='relu') 
        self.dense2 = tf.keras.layers.Dense(32, activation='relu') 
        self.output_layer = tf.keras.layers.Dense(num_actions) 

    def call(self, state): 
        x = self.dense1(state) 
        x = self.dense2(x) 
        return self.output_layer(x) 

q_network = QNetwork(num_actions)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.MeanSquaredError()

epsilon = 0.1
epsilon_decay = 0.99

num_episodes = 1000
gamma = 0.99

for episode in range(num_episodes): 
    state = np.random.random(num_states) 
    done = False 

    while not done: 
        if np.random.rand() < epsilon: 
            action = np.random.randint(num_actions) 
        else: 
            q_values = q_network(tf.expand_dims(state, axis=0)) 
            action = np.argmax(q_values.numpy()) 

        next_state = np.random.random(num_states) 
        reward = np.random.rand() 

        with tf.GradientTape() as tape: 
            q_values = q_network(tf.expand_dims(state, axis=0)) 
            target_q_values = q_values.numpy() 
            target_q_values[0, action] = reward + gamma * np.max(
                q_network(tf.expand_dims(next_state, axis=0)).numpy()
            )
            loss = loss_function(q_values, target_q_values)

        gradients = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

        state = next_state 

        if np.random.rand() < 0.1:
            done = True

    epsilon *= epsilon_decay

print(f"Episode {episode + 1}: Exploration Rate = {epsilon:.4f}")

# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 7.
# ---------------------------------------------------------------------------------------------------------------------
import warnings 
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
from sklearn.metrics import accuracy_score 

warnings.filterwarnings("ignore") 

iris = load_iris() 
X, y = iris.data, iris.target 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = MLPClassifier(
    hidden_layer_sizes=(5,),
    max_iter=1000,
    random_state=42,
    verbose=True
)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 8.
# ---------------------------------------------------------------------------------------------------------------------
import warnings 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_classification 
from keras.models import Sequential 
from keras.layers import LSTM, Dense 

warnings.filterwarnings("ignore")

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

X = X.reshape((X.shape[0], 1, X.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy}")

# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 9.
# ---------------------------------------------------------------------------------------------------------------------
import warnings 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_digits 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from keras.utils import to_categorical 

warnings.filterwarnings("ignore")

digits = load_digits() 
X, y = digits.images, digits.target 

X = X.reshape((X.shape[0], 8, 8, 1)) 
X = X / 16.0 
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy}")

# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 10.
# ---------------------------------------------------------------------------------------------------------------------
import warnings 
import numpy as np 
from sklearn.model_selection import train_test_split 
from keras.models import Model 
from keras.layers import Input, Flatten, Dense, Lambda 
from keras.datasets import mnist 
from keras import backend as K 

warnings.filterwarnings("ignore")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)

def siamese_network(input_shape):
    input_layer = Input(shape=input_shape)
    flatten = Flatten()(input_layer)
    dense_1 = Dense(128, activation='relu')(flatten)
    output_layer = Dense(32, activation='sigmoid')(dense_1)
    return Model(inputs=input_layer, outputs=output_layer)

input_shape = (28, 28)

siamese_branch = siamese_network(input_shape)

left_input = Input(input_shape)
right_input = Input(input_shape)

encoded_left = siamese_branch(left_input)
encoded_right = siamese_branch(right_input)

distance = Lambda(lambda x: K.abs(x[0] - x[1]))([encoded_left, encoded_right])

output_layer = Dense(1, activation='sigmoid')(distance)

siamese_model = Model(inputs=[left_input, right_input], outputs=output_layer)

siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

siamese_model.fit(
    [x_train[:, :, :, np.newaxis], x_train[:, :, :, np.newaxis]],
    y_train,
    epochs=5,
    batch_size=64,
    verbose=2
)

loss, accuracy = siamese_model.evaluate(
    [x_val[:, :, :, np.newaxis], x_val[:, :, :, np.newaxis]],
    y_val,
    verbose=0
)

print(f"Validation Accuracy: {accuracy}")