# ====================================================================================
# Combined Machine Learning & Deep Learning Code Snippets
# ====================================================================================

# ------------------------------------------------------------------------------------
# Snippet 1: Neural Network with Dropout
# ------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

# Define neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Dropout layer with 20% dropout rate
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Generate dummy data
x_train = np.random.rand(1000, 20)
y_train = np.random.randint(10, size=(1000,))

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(x_train, y_train)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# ------------------------------------------------------------------------------------
# Snippet 2: LSTM with Weight Clipping (MNIST)
# ------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Define LSTM model with weight clipping
model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(
        64,
        kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1.0)
    ),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model_lstm.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model_lstm.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# Evaluate the model
loss_lstm, accuracy_lstm = model_lstm.evaluate(x_test, y_test)
print(f"LSTM Model - Loss: {loss_lstm}, Accuracy: {accuracy_lstm}")

# ------------------------------------------------------------------------------------
# Snippet 3: GRU with Weight Clipping (MNIST)
# ------------------------------------------------------------------------------------

# Define GRU model with weight clipping
model_gru = tf.keras.Sequential([
    tf.keras.layers.GRU(
        64,
        kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1.0)
    ),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model_gru.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model_gru.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# Evaluate the model
loss_gru, accuracy_gru = model_gru.evaluate(x_test, y_test)
print(f"GRU Model - Loss: {loss_gru}, Accuracy: {accuracy_gru}")

# ------------------------------------------------------------------------------------
# Snippet 4: SGD Classifier on Iris Dataset
# ------------------------------------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Combine scaled data
combined_scaled_data = np.vstack((X_train_scaled, X_test_scaled))

# Create DataFrame
df = pd.DataFrame(
    combined_scaled_data,
    columns=iris.feature_names
)

print(df.head())

# Define and train SGD model
sgd_model = SGDClassifier(
    loss='log_loss',
    max_iter=1000,
    tol=1e-3,
    learning_rate='constant',
    eta0=0.01,
    shuffle=True
)

sgd_model.fit(X_train_scaled, y_train)

# Evaluate
sgd_score = sgd_model.score(X_test_scaled, y_test)
print("Accuracy (Stochastic Gradient Descent):", sgd_score)

# ------------------------------------------------------------------------------------
# Snippet 5: Mini-Batch Gradient Descent and Batch Gradient Descent
# ------------------------------------------------------------------------------------

from sklearn.linear_model import SGDClassifier

# Mini-batch model
mini_batch_model = SGDClassifier(
    loss='log_loss',
    max_iter=1,
    tol=1e-3,
    learning_rate='constant',
    eta0=0.01,
    shuffle=True
)

# Train on mini-batches
for epoch in range(100):
    for i in range(0, len(X_train_scaled), 32):
        mini_batch_model.partial_fit(
            X_train_scaled[i:i + 32],
            y_train[i:i + 32],
            classes=np.unique(y)
        )

# Evaluate
mini_batch_score = mini_batch_model.score(X_test_scaled, y_test)
print("Accuracy (Mini-batch Gradient Descent):", mini_batch_score)

# Batch Gradient Descent
batch_model = SGDClassifier(
    loss='log_loss',
    max_iter=1000,
    tol=1e-3,
    learning_rate='constant',
    eta0=0.01
)

batch_model.fit(X_train_scaled, y_train)

batch_score = batch_model.score(X_test_scaled, y_test)
print("Accuracy (Batch Gradient Descent):", batch_score)

# ------------------------------------------------------------------------------------
# Snippet 6: Grid Search for SVM Hyperparameter Tuning
# ------------------------------------------------------------------------------------

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

# Create SVM classifier
svm = SVC()

# Grid search
grid_search = GridSearchCV(
    svm,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# ------------------------------------------------------------------------------------
# Snippet 7: RMSprop vs Adam Optimizers
# ------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        64,
        activation='relu',
        input_shape=(X_train.shape[1],)
    ),
    tf.keras.layers.Dense(3, activation='softmax')
])

# RMSprop optimizer
optimizer_rmsprop = tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    momentum=0.9
)

model.compile(
    optimizer=optimizer_rmsprop,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=0
)

loss_rmsprop, accuracy_rmsprop = model.evaluate(X_test, y_test)

print(f"RMSprop Model - Loss: {loss_rmsprop}, Accuracy: {accuracy_rmsprop}")

# Adam optimizer
optimizer_adam = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)

model.compile(
    optimizer=optimizer_adam,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=0
)

loss_adam, accuracy_adam = model.evaluate(X_test, y_test)

print(f"Adam Model - Loss: {loss_adam}, Accuracy: {accuracy_adam}")

# ------------------------------------------------------------------------------------
# Snippet 8: Xavier (Glorot) Weight Initialization
# ------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

# Generate random data
num_samples = 1000
input_dim = 10
output_dim = 5

X = np.random.randn(num_samples, input_dim)
y = np.random.randn(num_samples, output_dim)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        32,
        activation='relu',
        kernel_initializer='glorot_uniform',
        input_shape=(input_dim,)
    ),
    tf.keras.layers.Dense(
        output_dim,
        activation='linear',
        kernel_initializer='glorot_uniform'
    )
])

# Compile
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train
history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Summary
model.summary()

# ------------------------------------------------------------------------------------
# Snippet 9: Lasso and Ridge Regression
# ------------------------------------------------------------------------------------

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error

# Generate regression data
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=0.1,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Lasso Regression
alpha_lasso = 0.1
lasso = Lasso(alpha=alpha_lasso)
lasso.fit(X_train, y_train)

# Ridge Regression
alpha_ridge = 0.1
ridge = Ridge(alpha=alpha_ridge)
ridge.fit(X_train, y_train)

# Predictions
y_pred_lasso = lasso.predict(X_test)
y_pred_ridge = ridge.predict(X_test)

# Mean Squared Error
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print("Mean Squared Error (Lasso):", mse_lasso)
print("Mean Squared Error (Ridge):", mse_ridge)
