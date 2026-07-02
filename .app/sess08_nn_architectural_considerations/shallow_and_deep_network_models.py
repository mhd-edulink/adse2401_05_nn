"""
=================================================================================================
Python script to demonstrate crop yield prediction using shallow & deep networks
=================================================================================================
This program demonstrates maize yield prediction using Shallow and Deep Networks
The simulated dataset simulates the relationship between soil nutrients, weather conditions, and maize yield
using both linear and non-linear interactions


Requirements:
    !pip install numpy matplotlib pandas seaborn scikit-learn tensorflow keras
"""

# -----------------------------------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------------------------------
from __future__ import annotations

from gc import callbacks

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Model, layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

import warnings

# Suppress warnings for a cleaner output demo
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------------
# 1. Constants and Global Configuration
# -----------------------------------------------------------------------------------------------
RANDOM_SEED = 42
DATASET_FILENAME = "../files/african_maize_yeild_data.csv"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")


# -----------------------------------------------------------------------------------------------
# 2. Dataset Generation
# -----------------------------------------------------------------------------------------------
def generate_dataset(num_samples: int = 5000) -> pd.DataFrame:
    # Fertiliser constitution
    nitrogen = np.random.uniform(20, 120, num_samples)
    phosphorus = np.random.uniform(10, 60, num_samples)
    pottasium = np.random.uniform(30, 150, num_samples)

    # Weather
    rainfall = np.random.uniform(400, 1400, num_samples)
    temperature = np.random.uniform(18, 36, num_samples)

    # Soil organic percentage
    organic_matter = np.random.uniform(1.0, 5.0, num_samples)

    # **************************
    # Simulated Yield Model
    # **************************
    linear_yield = (
            .1 * nitrogen +
            0.15 * phosphorus +
            0.05 * pottasium
    )

    drought_factor = np.where(
        (rainfall < 600) & (temperature > 30),
        -15 * (36 - temperature),
        0
    )

    buffer_factor = (organic_matter * (rainfall / 200))
    temperature_efficiency = (
            20 * np.sin(
        (temperature - 18) / 18 * np.pi
    )
    )

    # Add random noise to make the dataset more realistic
    noise = np.random.normal(0, 2, num_samples)

    # Maize yield
    yield_bags = (15 + linear_yield + drought_factor + buffer_factor + temperature_efficiency + noise)
    yield_bags = np.clip(yield_bags, 2, 90)

    # Generate dataset to be returned
    dataset = pd.DataFrame({
        "Nitrogen_NPK": nitrogen,
        "Phosphorus_NPK": phosphorus,
        "Pottasium_NPK": pottasium,
        "Average_rainfall_mm": rainfall,
        "Average_temperature_mm": temperature,
        "Soil_Organic_Matter_Pct": organic_matter,
        "Maize_Yield_Bags_Per_Ha": yield_bags,
    })

    return dataset


# -----------------------------------------------------------------------------------------------
# 3. Dataset storage & retreival
# -----------------------------------------------------------------------------------------------
def save_dataset(dataset: pd.DataFrame, filename: str = DATASET_FILENAME) -> None:
    dataset.to_csv(filename, index=False)
    print(f"Dataset successfully saved as: {filename}")


def load_dataset(filename: str = DATASET_FILENAME) -> pd.DataFrame:
    if not Path(filename).exists():
        raise FileNotFoundError(f"Dataset file '{filename}' does not exist'")
    print(f"Dataset successfully loaded from: {filename}")

    return pd.read_csv(filename)


# -----------------------------------------------------------------------------------------------
# 4. Data Exploration
# -----------------------------------------------------------------------------------------------
def display_dataset_information(dataset: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"DATASET OVERVIEW")
    print("\n" + "=" * 70)

    print(dataset.head())
    print(f"\nDataset Shape:\n{dataset.shape}")
    print(f"\nSummary Statistics:\n{dataset.describe()}")
    print(f"Missing values:\n{dataset.isnull().sum()}")


# -----------------------------------------------------------------------------------------------
# 5. Correlation Heatmap
# -----------------------------------------------------------------------------------------------
def plot_correlation_heatmap(dataset: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, cmap="viridis", fmt=".2f")
    plt.title("Correlation Matrix of the African Maize Yield Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
# 6. Data preprocessing
# -----------------------------------------------------------------------------------------------
def preprocess_data(dataset: pd.DataFrame) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler
]:
    print("\nPreprocessing dataset...")
    X = dataset.drop(columns=["Maize_Yield_Bags_Per_Ha"]).values
    y = dataset["Maize_Yield_Bags_Per_Ha"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Dataset successfully preprocessed!")
    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")

    return (X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, scaler)


# -----------------------------------------------------------------------------------------------
# 7. Shallow Network
# -----------------------------------------------------------------------------------------------
def build_shallow_model(input_dimension: int) -> Model:
    model = models.Sequential([
        layers.Input(shape=(input_dimension,)),
        layers.Dense(8, activation="relu", name="Hidden_Layer_1"),
        layers.Dense(1, activation="linear", name="Output"),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# -----------------------------------------------------------------------------------------------
# 8. Deep network
# -----------------------------------------------------------------------------------------------
def build_deep_model(input_dimension: int) -> Model:
    model = models.Sequential([
        layers.Input(shape=(input_dimension,)),
        layers.Dense(128, activation="relu", name="Hidden_Layer_1"),
        layers.Dense(64, activation="relu", name="Hidden_Layer_2"),
        layers.Dense(32, activation="relu", name="Hidden_Layer_3"),
        layers.Dense(16, activation="relu", name="Hidden_Layer_4"),
        layers.Dense(1, activation="linear", name="Output"),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# -----------------------------------------------------------------------------------------------
# 9. Model visualisation
# -----------------------------------------------------------------------------------------------
def display_model_summary(model: Model, model_name: str) -> None:
    print("\n" + "=" * 70)
    print(f"{model_name.upper()}")
    print("\n" + "=" * 70)

    model.summary()


def save_model_architecture(model: Model, filename: str) -> None:
    try:
        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True, dpi=150)
    except Exception as error:
        print(f"\nArchitecture diagram could not be generated.\n{error}")


# -----------------------------------------------------------------------------------------------
# 10. Early Stopping
# -----------------------------------------------------------------------------------------------
def create_early_stopping() -> EarlyStopping:
    return EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)


# -----------------------------------------------------------------------------------------------
# 11. Model Training
# -----------------------------------------------------------------------------------------------
def train_model(model: Model, X_train: np.ndarray, y_train: np.ndarray, epoch: int = 100,
                batch_size: int = 32) -> tf.keras.callbacks.History:
    callback = create_early_stopping()

    history = model.fit(X_train, y_train, epochs=epoch, validation_split=0.2, batch_size=batch_size,
                        callbacks=[callback], verbose=1)

    return history


# -----------------------------------------------------------------------------------------------
# 12. Save model
# -----------------------------------------------------------------------------------------------
def save_trained_model(model: Model, filename: str) -> None:
    model.save(filename)
    print(f"\nModel saved to:\n{filename}")


# -----------------------------------------------------------------------------------------------
# 13. Model Evaluation
# -----------------------------------------------------------------------------------------------
def evaluate_model(model: Model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    predictions = model.predict(X_test, verbose=0).flatten()
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} OVERVIEW")
    print("\n" + "=" * 70)

    print(f"Mean Squared Error: {mse:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")
    print(f"Mean absolute error: {mae:.3f}")
    print(f"R2 Score: {r2:.3f}")

    return {
        "name": model_name,
        "predictions": predictions,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


# -----------------------------------------------------------------------------------------------
# 14. Loss Curves
# -----------------------------------------------------------------------------------------------
def plot_loss_convergence(shallow_history, deep_history) -> None:
    plt.figure(figsize=(12, 8))
    plt.plot(shallow_history.history["loss"], label="Shallow Training", linewidth=2)
    plt.plot(deep_history.history["loss"], label="Deep Training", linewidth=2)
    plt.plot(deep_history.history["val_loss"], label="Deep Validation", linewidth=2)
    plt.title("Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
# 15. MAE Curves
# -----------------------------------------------------------------------------------------------
def plot_mae_convergence(shallow_history, deep_history) -> None:
    plt.figure(figsize=(12, 8))
    plt.plot(shallow_history.history["mae"], label="Shallow Training", linewidth=2)
    plt.plot(deep_history.history["mae"], label="Deep Training", linewidth=2)
    plt.plot(deep_history.history["val_mae"], label="Deep Validation", linewidth=2)
    plt.title("MAE Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
# 16. Actual vs Predicted
# -----------------------------------------------------------------------------------------------

def plot_actual_vs_predicted(y_true: np.ndarray, predictions: np.ndarray, model_name: str) -> None:
    plt.figure(figsize=(12, 8))
    plt.scatter(y_true, predictions, alpha=0.5, edgecolor="black")
    minimum = min(y_true.min(), predictions.min())
    maximum = max(y_true.max(), predictions.max())
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--", linewidth=2)
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title(f"{model_name}: Actual vs Predicted Yield")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
# 17. Residual Analysis
# -----------------------------------------------------------------------------------------------
def plot_residual(
        y_true: np.ndarray,
        predictions: np.ndarray,
        model_name: str,
) -> None:
    residuals = y_true - predictions
    plt.figure(figsize=(12, 8))
    plt.scatter(predictions, residuals, alpha=0.5, edgecolor="black")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted Yield")
    plt.ylabel("Residuals")
    plt.title(f"{model_name}: Residuals")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
# 18. Residual Histograms
# -----------------------------------------------------------------------------------------------
def plot_residual_distribution(y_true: np.ndarray, predictions: np.ndarray, model_name: str) -> None:
    residuals = y_true - predictions
    plt.figure(figsize=(12, 8))
    sns.histplot(residuals, bins=30, kde=True, color="steelblue")
    plt.title(f"{model_name}: Residual Distribution")
    plt.xlabel("Prediction Error")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
# 19. Model Performance Comparision
# -----------------------------------------------------------------------------------------------
def compare_models(shallow_results: dict, deep_results: dict) -> None:
    names = [
        shallow_results["name"],
        deep_results["name"],
    ]

    mae = [shallow_results["mae"], deep_results["mae"]]
    rmse = [shallow_results["rmse"], deep_results["rmse"]]

    r2 = [shallow_results["r2"], deep_results["r2"]]

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(names, mae, color=["royalblue", "forestgreen"])
    axes[0].set_title("MAE")
    axes[1].bar(names, rmse, color=["royalblue", "forestgreen"])
    axes[1].set_title("RMSE")
    axes[2].bar(names, r2, color=["royalblue", "forestgreen"])
    axes[2].set_title("R2")
    plt.tight_layout()
    plt.show()

    comparision = pd.DataFrame({
        "Model": names,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print("\n" + "=" * 70)
    print("MODEL COMPARISION")
    print(comparision)
    print("\n" + "=" * 70)

    winner = (
        shallow_results["name"]
        if shallow_results["mae"] < deep_results["mae"]
        else deep_results["name"]
    )

    print(f"\nBest Performing Model: {winner}")


# -----------------------------------------------------------------------------------------------
# 20. Plot Prediction Comparision
# -----------------------------------------------------------------------------------------------
def plot_prediction_comparision(y_test: np.ndarray, shallow_predictions: np.ndarray,
                                deep_predictions: np.ndarray) -> None:
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, shallow_predictions, alpha=0.5, label="Shallow")
    plt.scatter(y_test, deep_predictions, alpha=0.5, label="Deep")

    minimum = min(
        y_test.min(),
        shallow_predictions.min(),
        deep_predictions.min()
    )

    maximum = max(
        y_test.max(),
        shallow_predictions.max(),
        deep_predictions.max()
    )

    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--", linewidth=2)
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title("Prediction Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------------
# 21. Main Execution Function
# -----------------------------------------------------------------------------------------------
def main() -> None:
    """
    Execute the complete African maize yield demonstration.

    Workflow
    --------
    1. Generate synthetic dataset.
    2. Save dataset to CSV.
    3. Reload dataset.
    4. Explore dataset.
    5. Preprocess data.
    6. Train shallow neural network.
    7. Train deep neural network.
    8. Evaluate both models.
    9. Produce visualisations.
    10. Save trained models.
    """

    print("=" * 80)
    print("AFRICAN MAIZE YIELD PREDICTION")
    print("SHALLOW VS DEEP NEURAL NETWORKS")
    print("=" * 80)

    #############################################################################
    # Step 1. Generate synthetic dataset
    #############################################################################
    print(f"Step 1. Generate synthetic dataset.")
    dataset = generate_dataset()
    save_dataset(dataset)

    #############################################################################
    # Step 2. Load dataset
    #############################################################################
    print(f"Step 2. Load dataset.")
    dataset = load_dataset()
    display_dataset_information(dataset)
    plot_correlation_heatmap(dataset)

    #############################################################################
    # Step 3. Preprocess dataset
    #############################################################################
    print("\nStep 3. Preprocess dataset.")
    (X_train_scaled, X_test_scaled, y_train, y_test, x_train, x_test, scaler) = preprocess_data(dataset)

    #############################################################################
    # Step 4. Build Neural Networks
    #############################################################################
    print("Step 4. Build Neural Networks.")
    shallow_model = build_shallow_model(input_dimension=X_train_scaled.shape[1])
    deep_model = build_deep_model(input_dimension=X_train_scaled.shape[1])

    #############################################################################
    # Step 5. Model Summary
    #############################################################################
    print("Step 5. Model Summary")
    display_model_summary(shallow_model, "SHALLOW NETWORK")
    display_model_summary(deep_model, "DEEP NETWORK")

    #############################################################################
    # Step 6. Save Model Architecture Images.
    #############################################################################
    save_model_architecture(shallow_model, "Shallow_network.png")
    save_model_architecture(deep_model, "Deep_network.png")

    #############################################################################
    # Step 7. Train Shallow Neural Network
    #############################################################################
    print("Step 7. Train shallow neural network.")
    shallow_history = train_model(shallow_model, X_train_scaled, y_train ,epoch=100)

    #############################################################################
    # Step 8. Train Deep Neural Network
    #############################################################################
    print("Step 8. Train deep neural network.")
    deep_history = train_model(deep_model, X_train_scaled, y_train, epoch=100)

    #############################################################################
    # Step 9. Evaluate Models
    #############################################################################
    print("Step 9. Evaluate both models.")
    shallow_results = evaluate_model(shallow_model, X_test_scaled, y_test, "Shallow Network")
    deep_results = evaluate_model(deep_model, X_test_scaled, y_test, "Deep Network")

    #############################################################################
    # Step 10. Save trained models
    #############################################################################
    print("Step 10. Save trained models.")
    save_trained_model(shallow_model, "shallow_network.keras")
    save_trained_model(deep_model, "deep_network.keras")

    #############################################################################
    # Step 11. Generate visualisations
    #############################################################################
    print("Step 11. Generate visualisations.")
    plot_loss_convergence(shallow_history, deep_history)
    plot_mae_convergence(shallow_history, deep_history)

    #############################################################################
    # Step 12. Model Actual Vs Predicted
    #############################################################################
    print("Step 12. Model Actual vs Predicted")
    plot_actual_vs_predicted(y_test, shallow_results["predictions"], "Shallow Network")
    plot_actual_vs_predicted(y_test, deep_results["predictions"], "Deep Network")

    #############################################################################
    # Step 13. Compare models
    #############################################################################
    print("Step 13. Compare models")
    plot_prediction_comparision(y_test, shallow_results["predictions"], deep_results["predictions"])

    #############################################################################
    # Step 14. Plot residuals
    #############################################################################
    print("Step 14. Plot Residuals")
    plot_residual(y_test, shallow_results["predictions"], "Shallow Network")
    plot_residual(y_test, deep_results["predictions"], "Deep Network")

    #############################################################################
    # Step 15. Plot Residual Distribution
    #############################################################################
    print("Step 14. Plot Residuals Distribution")
    plot_residual_distribution(y_test, shallow_results["predictions"], "Shallow Network")
    plot_residual_distribution(y_test, deep_results["predictions"], "Deep Network")

    #############################################################################
    # Step 16. Compare Models
    #############################################################################
    print("Step 16. Compare models")
    compare_models(shallow_results, deep_results)


# -----------------------------------------------------------------------------------------------
# 22. Run the script by invoking its main() function
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
