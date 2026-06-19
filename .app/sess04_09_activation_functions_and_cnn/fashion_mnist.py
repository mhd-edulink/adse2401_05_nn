"""
=================================================================================================
Python script to demonstrate recognition using a Convolutional Neural Network and activation
functions
=================================================================================================

Requirements:
    !pip install numpy matplotlib pandas seaborn scikit-learn

Machine Learning Workflow:
    1. Deep Learning Workflow:
    2. Load Fashion-MNIST dataset
    3. Validate dataset
    4. Explore dataset structure
    5. Preprocess image data
    6. Build CNN architecture
    7. Apply ReLU activation functions
    8. Train CNN model
    9. Evaluate model performance
    10. Generate predictions
    11. Visualise results
    12. Save output figures

Dataset:
    Fashion-MNIST

Input:
    files/rainfall_temp_vs_yield.csv

Outputs:
    results/fashion_sample_images.png
    results/fashion_training_history.png
    results/fashion_confusion_matrix.png
    results/prediction_example.png

Requirements:
    !pip install matplotlib numpy scikit-learn tensorflow keras
"""
from cProfile import label

# -----------------------------------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("module")

# -----------------------------------------------------------------------------------------------
# 1. Constants
# -----------------------------------------------------------------------------------------------
RESULTS_DIR: Path = Path("../files/results")
RANDOM_STATE: int = 42
IMAGE_HEIGHT: int = 28
IMAGE_WIDTH: int = 28
NUM_CLASSES: int = 10
EPOCHS: int = 8
BATCH_SIZE: int = 32

# Fashion-MNIST class labels
CLASS_NAMES: list[str] = [
    "T-Shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# -----------------------------------------------------------------------------------------------
# 2. Seedng for reproducibility
# -----------------------------------------------------------------------------------------------
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# -----------------------------------------------------------------------------------------------
# 3. Dateset loading and validation
# -----------------------------------------------------------------------------------------------
def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f"Loading Fashion-MNIST dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        print(f"Fashion-MNIST dataset loaded successfully!")
        return x_train, y_train, x_test, y_test
    except Exception as exc:
        raise RuntimeError("Failed to load Fashion-MNIST dataset."
                           "\nPlease check your internet connection or TensorFlow installation "
                           "and try again.") from exc


def validate_dataset(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
) -> None:
    print(f"Validating dataset...")
    if x_train is None or y_train is None or x_test is None or y_test is None:
        raise ValueError("Dataset arrays cant be None!")

    if x_train.size == 0 or x_test.size == 0:
        raise ValueError("Dataset (x_train) arrays must not be empty!")

    if x_train.shape[1:] != (IMAGE_HEIGHT, IMAGE_HEIGHT):
        raise ValueError(f"Training images must be {IMAGE_HEIGHT} x {IMAGE_HEIGHT} pixels!"
                         f"\nGot {x_train.shape[1:]} instead.")

    if x_test.shape[1:] != (IMAGE_HEIGHT, IMAGE_HEIGHT):
        raise ValueError(f"Training images must be {IMAGE_HEIGHT} x {IMAGE_HEIGHT} pixels!"
                         f"\nGot {x_test.shape[1:]} instead.")

    if len(y_train) != x_train.shape[0]:
        raise ValueError(
            f"Number of training labels doe not match number of training images"
        )

    if len(y_test) != x_test.shape[0]:
        raise ValueError(
            "Number of testing labels does not match number of testing images"
        )

    if not (np.all(y_train >= 0) and np.all(y_train < NUM_CLASSES)):
        raise ValueError(
            f"Training labels must be in the range [0, {NUM_CLASSES - 1}]!"
        )

    if not (np.all(y_test >= 0) and np.all(y_test < NUM_CLASSES)):
        raise ValueError(
            f"Test labels must be in the range [0, {NUM_CLASSES - 1}]!]"
        )

    # When all test / validation passes notify user
    print(f"Dataset validated successfully!\n")


def display_dataset_information(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray
) -> None:
    print("-" * 70)
    print("DATASET INFORMATION")
    print(f"Training images shape       : {x_train.shape}")
    print(f"Training labels shape       : {y_train.shape}")
    print(f"Testing images shape        : {x_test.shape}")
    print(f"Testing labels shape        : {y_test.shape}")
    print(f"Number of classes           : {NUM_CLASSES}")
    print(f"Image Dimensions            : {IMAGE_HEIGHT} x {IMAGE_WIDTH}")
    print(f"Pixel value range (before)  : {x_train.min()}, {x_train.max()}")
    print("-" * 70)


# -----------------------------------------------------------------------------------------------
# 4. Data preprocessing
# -----------------------------------------------------------------------------------------------
def preproces_data(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Preprocessing images...")
    # Normalise pixel values from [0, 255] to [0, 1]
    x_train_norm = x_train.astype('float32') / 255.0
    x_test_norm = x_test.astype('float32') / 255.0

    # Reshape to add channel dimension: (28, 28) -> (28, 28, 1)
    x_train_reshaped = x_train_norm.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

    # One-hot encode the labels
    y_train_encoded = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_encoded = to_categorical(y_test, num_classes=NUM_CLASSES)

    print(f"Preprocessing complete."
          f"\nTraining datashape: {x_train_reshaped.shape}"
          f"\nLabel shape: {y_train_encoded.shape}")

    return x_train_reshaped, y_train_encoded, x_test_norm, y_test_encoded,


# -----------------------------------------------------------------------------------------------
# 5. Visualization utilities
# -----------------------------------------------------------------------------------------------
def plot_sample_images(
        x_train: np.ndarray,
        y_train: np.ndarray,
        save_path: Path
) -> None:
    print(f"Generating sample images plot...")

    # Recover integer labels if one hot encoded
    if y_train.ndim > 1:
        labels_int = np.argmax(y_train, axis=1)
    else:
        labels_int = y_train

    plt.figure(figsize=(10, 10))
    for n in range(25):
        plt.subplot(5, 5, n + 1)
        # Remove channel dimension for display
        plt.imshow(x_train[n].squeeze(), cmap="gray")
        plt.title(CLASS_NAMES[labels_int[n]], fontsize=10)
        plt.suptitle("Fashion MNIST Sample Images", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=150)
        plt.close()

        # Path to where sample images have been saved / stored
        print(f"Sample images saved to {save_path}\n")


def plot_training_history(
        history: tf.keras.callbacks.History,
        save_path: Path
) -> None:
    print(f"Generating training history plot...")

    plt.figure(figsize=(8, 6))
    plt.plot(
        history.history['accuracy'],
        label='Training Accuracy',
        marker='o',
    )

    plt.plot(
        history.history['val_accuracy'],
        label='Validation Accuracy',
        marker='s',
    )

    plt.title('Model Training History', fontsize=14, fontweight="bold")
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend("Lower right")
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Training history saved to {save_path}\n")


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Path
) -> None:
    print(f"Generating confusion matrix plot...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Fashion MNIST', fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    plt.yticks(tick_marks, CLASS_NAMES, fontsize=8)

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for n in range(NUM_CLASSES):
        for a in range(NUM_CLASSES):
            plt.text(
                a, n, format(cm[n, a], 'd'),
                ha='center', va='center', fontsize=8,
                color="white" if cm[n, a] > thresh else "black"
            )

    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to {save_path}\n")


def display_prediction_examples(
        model: tf.keras.Model,
        x_test: np.ndarray,
        y_test_true: np.ndarray,
        save_path: Path
) -> None:
    print(f"Generating prediction examples plot...")
    # Obtain predictions as an integer class indices
    y_pred_probs = model.predict(x_test[:10], verbose=0)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    plt.figure(figsize=(15, 7))
    for idx in range(10):
        plt.subplot(2, 5, idx + 1)
        plt.imshow(x_test[idx].squeeze(), cmap="gray")
        actual_class = CLASS_NAMES[y_test_true[idx]]
        predicted_class = CLASS_NAMES[y_pred_labels[idx]]
        colour = "green" if y_test_true[idx] == y_pred_labels[idx] else "red"
        plt.title(
            f"Actual: {actual_class}\nPredicted: {predicted_class}",
            fontsize=9,
            color=colour
        )
        plt.axis('off')

    plt.suptitle("Prediction Examples (Green = Correct, Red = Incorrect)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()  # plt.show() to display the graph
    print(f"Prediction examples saved to {save_path}\n")


def predict_single_image(
        model: tf.keras.Model,
        image: np.ndarray,
        true_label: int
) -> None:
    # Expand dimensions to create a batch size 1
    image_batch = np.expand_dims(image, axis=0)
    prediction_probs = model.predict(image_batch, verbose=0)[0]
    predicted_class = np.argmax(prediction_probs)
    print("-" * 70)
    print(f"SINGLE IMAGE PREDICTION DEMONSTRATION")
    print("-" * 70)
    print(f"True label: {CLASS_NAMES[true_label]}")
    print(f"Predicted label:{CLASS_NAMES[predicted_class]}")
    print(f"Confidence score: {prediction_probs[predicted_class]}")
    print(f"Class probabilities: {prediction_probs}")
    for idx, prob in enumerate(prediction_probs):
        marker = " <--" if idx == predicted_class else ""
        print(f"{CLASS_NAMES[idx]:<15s}: {prob:.4f}{marker}")
    print("-" * 70)


# -----------------------------------------------------------------------------------------------
# 6. CNN model definition
# -----------------------------------------------------------------------------------------------
def build_cnn_model() -> Sequential:
    print("Building CNN model...")
    model = Sequential([
        # First convolution block
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation='relu',  # ReLU activation: introduces non-linearity to prevent vanishing gradient probelm
            padding='same',
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1),
        ),
        MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional block
        Conv2D(
            64,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
        ),
        MaxPooling2D(pool_size=(2, 2)),

        # Flattent the 2D feature maps into a 1D vector
        Flatten(),

        # Fully connected (dense) layer with ReLU activation
        Dense(128, activation='relu'),

        # Output layer with Softmax activation:
        # Converts raw scores into a probability distribution over the 10 classes
        Dense(NUM_CLASSES, activation='softmax'),

    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Model built and compiled successfully.\n")
    # Display a summary of the model architecture
    model.summary()
    print()
    return model


# ===============================================================================================
# 7. Model Training and Evaluation
# ===============================================================================================
def train_model(
        model: Sequential,
        x_train: np.ndarray,
        y_train: np.ndarray,
) -> tf.keras.callbacks.History:
    print("Training model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )
    print("Training complete.\n")
    return history


def evaluate_model(
        model: Sequential,
        x_test: np.ndarray,
        y_test: np.ndarray,
) -> tuple[float, float]:
    print("Evaluating model...")
    test_loss, test_accuraccy = model.evaluate(x_test, y_test, verbose=0)

    # Obtain integer predictions for classification report
    y_test_int = np.argmax(y_test, axis=1)
    y_pred_probs = model.predict(x_test[:10], verbose=0)
    y_pred_int = np.argmax(y_pred_probs, axis=1)

    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("\n" + "-" * 70)
    print(
        classification_report(
            y_test_int,
            y_pred_int,
            target_names=CLASS_NAMES,
        )
    )
    print("\n" + "-" * 70)
    return test_loss, test_accuraccy


# -----------------------------------------------------------------------------------------------
# 8. Main Execution Function
# -----------------------------------------------------------------------------------------------
def main() -> None:
    # Ensure the results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the dataset
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_dataset()

    # 2. Validate dataset integrity
    validate_dataset(x_train_raw, y_train_raw, x_test_raw, y_test_raw)

    # 3. Display dataset information
    display_dataset_information(x_train_raw, y_train_raw, x_test_raw, y_test_raw)

    # 4. Preprocess the data
    x_train_proc, y_train_proc, x_test_proc, y_test_proc = preproces_data(x_train_raw, y_train_raw, x_test_raw,
                                                                          y_test_raw)

    # 5. Build the CNN model
    model = build_cnn_model()

    # 6. Train the model
    history = train_model(model, x_train_proc, y_train_proc)

    # 7. Evaluate the model
    test_loss, test_accuracy = evaluate_model(model, x_test_proc, y_test_proc)

    # 8. Generate visualisations
    print("Generating visualisations...")

    # Sample images from the raw training set (before preprocessing for display)
    plot_sample_images(
        x_train_proc, y_train_proc,
        RESULTS_DIR / "fashion_sample_images.png",
    )

    # Training history plot
    plot_training_history(history, RESULTS_DIR / "fashion_training_history.png")

    # Confusion matrix
    y_test_int = np.argmax(y_test_proc, axis=1)
    y_pred_int = np.argmax(model.predict(x_test_proc, verbose=0), axis=1)
    plot_confusion_matrix(
        y_test_int,
        y_pred_int,
        RESULTS_DIR / "fashion_confusion_matrix.png",
    )

    # Prediction examples
    display_prediction_examples(
        model,
        x_test_proc,
        y_test_int,
        RESULTS_DIR / "fashion_prediction_examples.png",
    )

    print(f"All images / visualizations generated and saved")

    # 9. Single image prediction demonstration
    predict_single_image(
        model,
        x_test_proc[0],
        y_test_int[0],
    )

    # 10. Final output summary
    print("-" * 70)
    print("FASHION-MNIST CNN RESULTS")
    print("-" * 70)
    print("END OF DEMO")


# -----------------------------------------------------------------------------------------------
# 9. Run the script by invoking its main() function
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
