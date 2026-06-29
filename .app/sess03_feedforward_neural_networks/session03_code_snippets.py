# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 1: Forward Propagation + Visualization
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward propagation function
def forward_propagation(inputs, weights, biases):
    weighted_sum = np.dot(weights, inputs) + biases
    activation = sigmoid(weighted_sum)
    return activation, weighted_sum

inputs = np.array([0.5, 0.3, 0.2])

weights = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.8, 0.5]
])

biases = np.array([0.1, 0.3])

output, weighted_sum = forward_propagation(inputs, weights, biases)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(np.arange(len(inputs)), inputs, color='blue')
plt.title('Input Values')
plt.xlabel('Input Neuron')
plt.ylabel('Value')

plt.subplot(1, 2, 2)
plt.bar(np.arange(len(weighted_sum)), weighted_sum, color='green')
plt.title('Weighted Sum (Before Activation)')
plt.xlabel('Neuron')
plt.ylabel('Weighted Sum')

plt.tight_layout()
plt.show()

print("Output after forward propagation:", output)


# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 2: Neural Network from Scratch (Backpropagation)
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))

    def forward(self, inputs):
        self.hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_outputs = sigmoid(self.hidden_inputs)

        self.output_inputs = np.dot(self.hidden_outputs, self.weights_hidden_output) + self.bias_output
        self.output_outputs = sigmoid(self.output_inputs)

        return self.output_outputs

    def backward(self, inputs, targets, learning_rate):
        output_errors = targets - self.output_outputs
        output_delta = output_errors * sigmoid_derivative(self.output_outputs)

        hidden_errors = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_errors * sigmoid_derivative(self.hidden_outputs)

        self.weights_hidden_output += np.dot(self.hidden_outputs.T, output_delta) * learning_rate
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate


# Training data
np.random.seed(0)
X = np.random.rand(100, 2)
y = np.array([int(x1 + x2 > 1) for x1, x2 in X])

nn = NeuralNetwork(2, 3, 1)

losses = []

for epoch in range(1000):
    outputs = nn.forward(X)
    nn.backward(X, y.reshape(-1, 1), 0.1)

    loss = np.mean((y.reshape(-1, 1) - outputs) ** 2)
    losses.append(loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 3: Perceptron
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, max_epochs=100):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for epoch in range(self.max_epochs):
            errors = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction

                if error != 0:
                    self.weights += self.learning_rate * error * inputs
                    self.bias += self.learning_rate * error
                    errors += 1

            if errors == 0:
                print(f"Converged at epoch {epoch}")
                break
            elif epoch == self.max_epochs - 1:
                print("Max epochs reached. Perceptron could not converge.")


training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 1])

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)

for test_input in training_inputs:
    print(f"{test_input} --> {perceptron.predict(test_input)}")


# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 4: TensorFlow MNIST Neural Network
# ---------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap='gray')
    plt.xlabel(y_train[i])

plt.show()

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test)
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

for i in range(10):
    print("Predicted:", predicted_labels[i])
    print("Actual:", y_test[i])
    print()