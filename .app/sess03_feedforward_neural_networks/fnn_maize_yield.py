"""
=================================================================================================
Python script to demonstrates FeedForward Neural Network for Maize yield prediction
=================================================================================================
This program demonstrates how rainfall and temperature data can be used to predict maize yeild
using a Feedforward Neural Network.


Requirements:
    !pip install numpy matplotlib pandas seaborn scikit-learn
"""

# -----------------------------------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------------
# 1. Constants
# -----------------------------------------------------------------------------------------------
DATA_FILE = Path("../files/rainfall_temp_vs_yield.csv")
RESULTS_DIR = Path("../files/results")
RANDOM_STATE = 42

REQUIRED_COLUMNS = [
    "ID",
    "Rain_M1",
    "Temp_M1",
    "Rain_M2",
    "Temp_M2",
    "Rain_M3",
    "Temp_M3",
    "Rain_M4",
    "Temp_M4",
    "Yield"
]


# -----------------------------------------------------------------------------------------------
# 2. Functions
# -----------------------------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"File not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        raise ValueError("Dataset is empty")

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df["Yield"].isna().all():
        raise ValueError("Yeild column contains no value")


def create_correlation_matrix(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 10))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "correlation_matrix.png", dpi=300)

    plt.close()


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    df["Total_Rainfall"] = (
            df["Rain_M1"] +
            df["Rain_M2"] +
            df["Rain_M3"] +
            df["Rain_M4"]
    )

    df["Average_Temperature"] = df[
        ["Temp_M1", "Temp_M2", "Temp_M3", "Temp_M4"]
    ].mean(axis=1)

    feature_columns = [
        "Rain_M1",
        "Temp_M1",
        "Rain_M2",
        "Temp_M2",
        "Rain_M3",
        "Temp_M3",
        "Rain_M4",
        "Temp_M4",
        "Total_Rainfall",
        "Average_Temperature"
    ]

    x = df[feature_columns]
    y = df["Yield"]

    return x, y


def build_model() -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_STATE
    )


def train_model(
        model: MLPRegressor,
        x_train: np.ndarray,
        y_train: pd.Series,
        x_validation: np.ndarray,
        y_validation: pd.Series,
        epochs: int = 200
)-> tuple[MLPRegressor, list[float], list[float]]:

    train_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.fit(x_train, y_train)

        train_predictions = model.predict(x_train)
        validation_predictions = model.predict(x_validation)

        train_loss = mean_squared_error(y_train, train_predictions)
        validation_loss = mean_squared_error(y_validation, validation_predictions)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if(epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Validation Loss: {validation_loss:.4f}"
            )

    return model, train_losses, validation_losses


def evaluate_model(
        model: MLPRegressor,
        x_test_scaled: np.ndarray,
        y_test: pd.Series,
) -> tuple:
    predictions = model.predict(x_test_scaled)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2, predictions


def plot_training_history(
        train_losses: list[float],
        validation_losses: list[float],
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")

    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "training_history.png", dpi=300)
    plt.close()  # plt.show() to display
    print(train_losses)
    print(validation_losses)
    print(len(train_losses))
    print(len(validation_losses))


def plot_actual_vs_predicted(
        actual: pd.Series,
        predicted: pd.Series,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.8)

    minimum = min(actual.min(), predicted.min())
    maximum = min(actual.max(), predicted.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--"
    )

    plt.title("Actual vs Predicted Yield")
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=300)
    plt.close()  # plt.show() to display


def plot_residuals(
        actual,
        predicted,
        alpha=0.8,
) -> None:

    residuals = actual - predicted

    plt.figure(figsize=(8, 6))

    plt.scatter(
        predicted,
        residuals,
        alpha=alpha,
    )

    plt.axhline(
        y=0,
        linestyle="--",
        color="red",
    )

    plt.title("Residual Plot")
    plt.xlabel("Predicted Yield")
    plt.ylabel("Residuals")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "residuals.png", dpi=300)
    plt.close()

# -----------------------------------------------------------------------------------------------
# 2. Main Execution Function
# -----------------------------------------------------------------------------------------------
def main() -> None:
    try:
        RESULTS_DIR.mkdir(exist_ok=True, parents=True)

        print("Loading dataset...")
        df = load_dataset()

        print("Validating dataset...")
        validate_dataset(df)

        print("Handling missing values...")
        df = df.dropna()

        print("Creating correlation matrix...")
        create_correlation_matrix(df)

        print("Preparing features...")
        x, y = prepare_data(df)

        print("Splitting dataset...")
        x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=RANDOM_STATE)

        x_validation, x_test, y_validation, y_test = (
            train_test_split(
                x_temp,
                y_temp,
                test_size=0.5,
                random_state=RANDOM_STATE
            )
        )

        print("Scaling features...")
        scaler = StandardScaler()

        x_train_scaled = scaler.fit_transform(x_train)
        x_validation_scaled = scaler.transform(x_validation)
        x_test_scaled = scaler.transform(x_test)

        print("Building neural network...")
        model = build_model()

        print("Training neural network...")
        model, train_losses, validation_losses = (train_model(
            model,
            x_train_scaled,
            y_train,
            x_validation_scaled,
            y_validation,
            epochs=200
        ))

        print("Evaluating model...")
        mae, rmse, r2, predictions = evaluate_model(model, x_test_scaled, y_test)

        print("Generating visualisations...")
        print(f"TESTING DATA FOR PLOTTING HISTORY: {train_losses}")
        print(f"TESTING DATA FOR PLOTTING HISTORY: {validation_losses}")
        plot_training_history(train_losses, validation_losses)
        plot_actual_vs_predicted(y_test, predictions)
        plot_residuals(y_test, predictions)

        print("\n-----------------------------------------------------------------------")
        print("MAIZE YEILD PREDICTIONS RESULTS")
        print("\n-----------------------------------------------------------------------")

        print(f"Training Records: {len(x_train)}")
        print(f"Validation Records: {len(x_validation)}")
        print(f"Testing Records: {len(x_test)}")

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Output files saved to: {RESULTS_DIR.resolve()}")
        print("\nEND OF DEMONSTRATION")

    except Exception as e:
        print(f"\nERROR: {e}")


# -----------------------------------------------------------------------------------------------
# 3. Run the script by invoking its main() function
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
