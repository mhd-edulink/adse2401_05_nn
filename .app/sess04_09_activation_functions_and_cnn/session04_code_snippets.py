# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 1: Activation Functions Comparison (Linear, Sigmoid, ReLU, Tanh)
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 1000)

y_linear = linear(x)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

plt.figure(figsize=(10, 6))

plt.plot(x, y_linear, label='Linear', color='blue')
plt.plot(x, y_sigmoid, label='Sigmoid', color='green')
plt.plot(x, y_relu, label='ReLU', color='red')
plt.plot(x, y_tanh, label='Tanh', color='orange')

plt.title('Linear & Non-Linear Activation Functions', fontsize=20)
plt.xlabel('Input', fontsize=14)
plt.ylabel('Output', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 2: Sigmoid Activation Function Plot
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x, y)
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 3: Tanh Activation Function Plot
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

x = np.linspace(-10, 10, 100)
y = tanh(x)

plt.plot(x, y)
plt.title('Hyperbolic Tangent (tanh) Activation Function')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True)
plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 4: ReLU Activation Function Plot
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-10, 10, 100)
y = relu(x)

plt.plot(x, y)
plt.title('Rectified Linear Unit (ReLU) Activation Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid(True)
plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 5: ReLU Variants (Leaky ReLU, Swish, GELU)
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def swish(x):
    return x * (1 / (1 + np.exp(-x)))

def gelu(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

x = np.linspace(-10, 10, 100)

y_leaky_relu = leaky_relu(x)
y_swish = swish(x)
y_gelu = gelu(x)

plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.plot(x, y_swish, label='Swish')
plt.plot(x, y_gelu, label='GELU')

plt.title('ReLU Variants')
plt.xlabel('x')
plt.ylabel('Activation(x)')
plt.legend()
plt.grid(True)
plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 6: ELU Activation Function
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha=1.0):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)

x = np.linspace(-5, 5, 100)
y = elu(x)

plt.plot(x, y)
plt.title('Exponential Linear Unit (ELU) Activation Function')
plt.xlabel('x')
plt.ylabel('ELU(x)')
plt.grid(True)
plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Code Snippet 7: Softmax Activation Function
# ---------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores)

logits = np.array([2.0, 1.0, 0.1])
softmax_probs = softmax(logits)

plt.bar(range(len(softmax_probs)), softmax_probs)
plt.title('Softmax Activation Function')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.xticks(range(len(softmax_probs)))
plt.grid(axis='y')
plt.show()