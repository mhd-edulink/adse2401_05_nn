"""
=================================================================================================
Python script to demonstrates Artificial Neural Network
=================================================================================================
This program implements a simple feed-forward neural network with one hidden layer, trained for
manual forward propogation, backward propogation, and gradient descent, to learn the XOR function
features:
    - 1.  Text loading and corpus management
    - 2.  Tokenisation and text cleaning
    - 3.  Vocabulary construction with word-to-index mappings
    - 4.  Sliding-window training sequence generation
    - 5.  Word embeddings
    - 6.  Neural language modelling with PyTorch
    - 7.  Next-word prediction
    - 8.  Temperature-controlled text generation

Requirements:
    !pip install numpy matplotlib
"""

# -----------------------------------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------------------------
# 1. Sigmoid function and sigmoid derivative
# -----------------------------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Calculate the sigmoid activation value.

    Parameters
    ----------
    x : np.ndarray
        Input values supplied to the activation function.

    Returns
    -------
    np.ndarray
        Values transformed into the range 0 to 1.

    Notes
    -----
    The sigmoid function introduces non-linearity into the neural
    network. Without non-linear activation functions, a multi-layer
    network would behave like a single linear transformation,
    regardless of how many layers it has. Sigmoid is widely used in
    educational demonstrations because its smooth, S-shaped curve and
    simple derivative make it easy to reason about during
    backpropagation.
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1.0 - x)


# -----------------------------------------------------------------------------------------------
# 2. Neural Network class
# -----------------------------------------------------------------------------------------------
class NeuralNetwork:
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            learning_rate: float,
            seed: int = 42
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        rng = np.random.default_rng(seed)

        # Weights are initialised randomly (small values) so that
        # neurons start with different parameters and can learn
        # distinct features during training
        self.weights_input_hidden: np.ndarray = rng.uniform(
            -1.0,
            1.0,
            size=(input_size, hidden_size)
        )

        self.bias_hidden: np.ndarray = np.zeros((1, hidden_size))
        self.weights_hidden_output: np.ndarray = rng.uniform(
            -1.0, 1.0, size=(self.hidden_size, output_size)
        )
        self.bias_output: np.ndarray = np.zeros((1, output_size))

        # Placeholders that will store intermediate values computed
        # during forward propagation. These are required again during
        # backward propgation, so they are stored as attributes
        self.hidden_weighted_sum: np.ndarray = np.zeros((1, hidden_size))
        self.hidden_activation: np.ndarray = np.zeros((1, hidden_size))
        self.output_weighted_sum: np.ndarray = np.zeros((1, output_size))

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        # Hidden layer computations
        self.hidden_weighted_sum = (inputs @ self.weights_input_hidden + self.bias_hidden)
        self.hidden_activation = sigmoid(self.hidden_weighted_sum)

        # Output layer computations
        self.output_weighted_sum = sigmoid(
            self.hidden_activation @ self.weights_hidden_output + self.bias_output
        )
        self.output_activation = sigmoid(self.output_weighted_sum)

        return self.output_activation

    @staticmethod
    def calculate_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean(targets - predictions) ** 2)

    def backward_propagation(
            self, inputs: np.ndarray, targets: np.ndarray
    ) -> None:
        """
        Perform backward propagation and update the network's
        weights and biases using gradient descent.

        Parameters
        ----------
        inputs : np.ndarray
            The input data used in the most recent forward pass,
            of shape (n_samples, input_size).
        targets : np.ndarray
            The true target values corresponding to ``inputs``,
            of shape (n_samples, output_size).

        Returns
        -------
        None

        Notes
        -----
        Backward propagation applies the chain rule to compute how
        much each weight and bias contributed to the overall error,
        then adjusts each parameter slightly in the direction that
        reduces the error (gradient descent).

        The steps are as follows:

        1. Output layer error:
           ``error_output = targets - output_activation``

        2. Output layer delta (how much to adjust the output
           weighted sum):
           ``delta_output = error_output * sigmoid_derivative(output_activation)``

        3. Hidden layer error (error propagated backwards through the
           output weights):
           ``error_hidden = delta_output @ W_hidden_output.T``

        4. Hidden layer delta:
           ``delta_hidden = error_hidden * sigmoid_derivative(hidden_activation)``

        5. Weight and bias gradients are computed from the deltas and
           the activations/inputs that fed into each layer.

        6. Gradient descent update rule:
           ``parameter = parameter + learning_rate * gradient``

           (Addition is used because ``error_output`` is defined as
           ``targets - predictions``, which already points in the
           direction that reduces the loss.)
        """
        n_samples = inputs.shape[0]

        # --- Step 1: Output layer error ---
        # How far off were our predictions from the true targets?
        error_output = targets - self.output_activation

        # --- Step 2: Output layer delta ---
        # Scale the error by the sensitivity (derivative) of the
        # output activation with respect to its weighted sum.
        delta_output = error_output * sigmoid_derivative(
            self.output_activation
        )

        # --- Step 3: Hidden layer error ---
        # Propagate the output delta backwards through the
        # output-layer weights to find each hidden neuron's
        # contribution to the final error.
        error_hidden = delta_output @ self.weights_hidden_output.T

        # --- Step 4: Hidden layer delta ---
        # Scale the hidden error by the sensitivity (derivative) of
        # the hidden activation with respect to its weighted sum.
        delta_hidden = error_hidden * sigmoid_derivative(
            self.hidden_activation
        )

        # --- Step 5: Compute gradients for weights and biases ---
        # Gradients are averaged over the batch (divided by
        # n_samples) so that the learning rate behaves consistently
        # regardless of how many training examples are used.
        grad_weights_hidden_output = (
                                             self.hidden_activation.T @ delta_output
                                     ) / n_samples
        grad_bias_output = (
                np.sum(delta_output, axis=0, keepdims=True) / n_samples
        )

        grad_weights_input_hidden = (inputs.T @ delta_hidden) / n_samples
        grad_bias_hidden = (
                np.sum(delta_hidden, axis=0, keepdims=True) / n_samples
        )

        # --- Step 6: Gradient descent weight and bias updates ---
        # Each parameter is nudged in the direction that reduces the
        # error, scaled by the learning rate.
        self.weights_hidden_output += (
                self.learning_rate * grad_weights_hidden_output
        )
        self.bias_output += self.learning_rate * grad_bias_output

        self.weights_input_hidden += (
                self.learning_rate * grad_weights_input_hidden
        )
        self.bias_hidden += self.learning_rate * grad_bias_hidden

    def train(
            self,
            inputs: np.ndarray,
            targets: np.ndarray,
            epochs: int,
            progress_interval: int = 500,
    ) -> list[float]:

        loss_history: list[float] = []

        for epoch in range(1, epochs + 1):
            predictions = self.forward_propagation(inputs)
            loss = self.calculate_loss(predictions, targets)
            loss_history.append(loss)

            self.backward_propagation(inputs, targets)

            if epoch % progress_interval == 0 or epoch == 1:
                print(f"Epoch {epoch:<5} | Loss: {loss:.4f}")

        return loss_history


def print_section_heading(title: str) -> None:
    line = "=" * 40
    print(f"\n{line}\n{title}\n{line}")


def print_network_parameters(network: NeuralNetwork) -> None:
    print("Weights (Input -> Hidden): ")
    print(network.weights_input_hidden)

    print("Biases (Hidden Layer):")
    print(network.bias_hidden)

    print("Weights (Hidden Output):")
    print(network.weights_hidden_output)

    print("Bias (Output Layer):")
    print(network.bias_output)


def print_predictions(
        network: NeuralNetwork,
        inputs: np.ndarray,
        targets: np.ndarray,
) -> None:
    predictions = network.forward_propagation(inputs)

    for n in range(inputs.shape[0]):
        print(f"Input: {inputs[n]}")
        print(f"Target: {int(targets[n,0])}")
        print(f"Prediction: {predictions[n,0]:.3f}\n")


def plot_sigmoid_curve() -> None:
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="Sigmoid (x)", color="blue")
    plt.title("Sigmoid Activation Function")
    plt.xlabel("x")
    plt.ylabel("sigmoid (x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss_curve(loss_history: list[float]) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color="red")
    plt.title("Training Loss Over Epocs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (Loss)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
# 3. Main Execution Function
# -----------------------------------------------------------------------------------------------
def main() -> None:
    # --- XOR dataset ---
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )

    y = np.array(
        [
            [0],
            [1],
            [1],
            [0],
        ],
        dtype=float,
    )

    # --- Hyperparameters ---
    epochs = 5000
    learning_rate = 0.5

    # --- Create the network ---
    network = NeuralNetwork(
        input_size=2,
        hidden_size=2,
        output_size=1,
        learning_rate=learning_rate,
        seed=42
    )

    # ---  ---
    print_section_heading("NETWORK ARCHITECTURE")
    print("Input Layer  :  2 neurons")
    print("Hidden Layer :  2 neurons (sigmoid activation)")
    print("Output Layer :  1 neuron  (sigoid activation)")

    print(f"Learning Rate:  {learning_rate}")
    print(f"Epochs:  {epochs}")

    # --- Display initial parameters ---
    print_section_heading("INITIAL NETWORK PARAMETERS")
    print_network_parameters(network)

    # --- Train the network ---
    print_section_heading("TRAINING PROGRESS")
    loss_history = network.train(X, y, epochs, progress_interval=500)

    # --- Display final paramters ---
    print_section_heading("FINAL NETWORK PARAMETERS")
    print_network_parameters(network)

    # --- Display final prediction ---
    print_section_heading("FINAL PREDICTIONS")
    print_predictions(network, X, y)

    # --- Visualisations ---
    print_section_heading("GENERATING VISUALIZATION")
    print("1. Sigmoid activation function curve...")
    print("2. Training loss curve...")
    plot_sigmoid_curve()


# -----------------------------------------------------------------------------------------------
# 4. Run the script by invoking its main() function
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
