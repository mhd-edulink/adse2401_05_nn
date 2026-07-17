"""
=======================================================================================
Python script to demonstrate Recurrent Neural Networks to predict stock prices
=======================================================================================

This program demonstrates RNN for stock prediction using TensorFlow & Keras.
It showcased how Recurrent nns learn temporel patterns within financial time series data.

The network predicts the next closing stock price using historical prices obtained
from Yahoo Finance

Learning outcomes

1. How sequential data differs from conventional datasets.
2. Why recurrent neural networks are suitable for time series.
3. How historical observations are transformed into input sequences.
4. How stock price prediction is performed using an RNN.
5. How regression models are evaluated using common error metrics.

Requirements:
    matplotlib, numpy, yfinance, scikit-learn, tensorflow, keras
"""

# -----------------------------------------------------------------------
# 0. Import required modules
# -----------------------------------------------------------------------
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from keras.layers import Dense, SimpleRNN
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# 1. Function to load the dataset
# -----------------------------------------------------------------------------
def load_dataset(
        ticker: str = "NVDA", start_date: str = "2020-01-01", end_date: str = "2025-10-01"
) -> tuple:
    data = yf.download(ticker, start=start_date, end=end_date)
    # SAFETY CHECK: Catch empty data before it reaches the scaler
    if data.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}' between {start_date} and {end_date}. "
            f"Please check the ticker symbol, your date range, or your internet connection."
        )
    # closing_prices = data["Close"].values.reshape(-1, 1)
    closing_prices = data["Close"].values
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(closing_prices)
    return scaled_prices, scaler


# -----------------------------------------------------------------------------
# 2. Function to create input sequences for time series prediction
# -----------------------------------------------------------------------------
def create_sequences(data: np.ndarray, sequence_length=60) -> tuple:
    x_data = []
    y_data = []

    for index in range(len(data) - sequence_length):
        x_data.append(
            data[
                index: index + sequence_length,
                0,
            ]
        )
        y_data.append(
            data[
                index + sequence_length,
                0,
            ]
        )

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    x_data = x_data.reshape(
        x_data.shape[0],
        x_data.shape[1],
        1,
    )
    return x_data, y_data


# -----------------------------------------------------------------------------
# 3. Function to split the dataset into training and testing sets
# -----------------------------------------------------------------------------
def split_dataset(
        x_data: np.ndarray, y_data: np.ndarray, train_size: float = 0.8
) -> tuple:
    split_index = int(len(x_data) * train_size)
    x_train = x_data[:split_index:]
    x_test = x_data[split_index:]

    y_train = y_data[:split_index]
    y_test = y_data[split_index:]
    return x_train, y_train, x_test, y_test


# -----------------------------------------------------------------------------
# 4. Function to construct a simple recurrent nn
# -----------------------------------------------------------------------------
def build_rnn(sequence_length: int) -> Sequential:
    model = Sequential()

    model.add(SimpleRNN(50, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(SimpleRNN(50))

    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model


# -----------------------------------------------------------------------------
# 5. Function to train the RNN Model
# -----------------------------------------------------------------------------
def train_model(model, x_train, y_train):
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1)
    return history


# -----------------------------------------------------------------------------
# 6. Function to evalutae the RNN Model
# -----------------------------------------------------------------------------
def evalutae_model(model, x_test, y_test, scaler):
    predictions = scaler.inverse_transform(model.predict(x_test, verbose=0))
    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(actual_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, predictions)

    # display mse, rmse, and mae
    print(
        f"Mean Squared Error: {mse:.2f}"
        f"Mean Absolute Error: {mae:.2f}"
        f"Root Mean Squared Error: {rmse:.2f}"
    )

    return actual_values, predictions


# -----------------------------------------------------------------------------
# 7. Function to visualise actual and predicted prices
# -----------------------------------------------------------------------------
def visualise_results(actual_values, predictions, ticker):
    plt.figure(figsize=(12, 8))
    plt.plot(actual_values, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Trading days")
    plt.ylabel("Closing price")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 8. Main execution function
# -----------------------------------------------------------------------------
def main() -> None:
    ticker = "NVDA"

    print("=" * 70)
    print("Recurrent Neural Networks Demonstration")
    print("=" * 70)
    print("Application: Stock Price Prediction")
    print(f"Ticker                    :{ticker}")
    print("Architecture               :Simple RNN")
    print("=" * 70)

    scaled_prices, scaler = load_dataset(ticker=ticker)
    x_data, y_data = create_sequences(scaled_prices)

    # x_train, x_test, y_train, y_test = split_dataset(x_data, y_data)
    x_train, y_train, x_test, y_test = split_dataset(x_data, y_data)
    model = build_rnn(
        x_train.shape[1],
    )
    model.summary()

    train_model(model, x_train, y_train)

    actual_values, predictions = evalutae_model(model, x_test, y_test, scaler)
    visualise_results(actual_values, predictions, ticker)


# -----------------------------------------------------------------------------
# 9. Run the script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
