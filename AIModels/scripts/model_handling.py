import os
import torch
import pandas as pd
import numpy as np
from scripts.modules.train_models import train_lstm_model, save_model_and_scaler, load_model_and_scaler
from scripts.modules.data_handling import fetch_process_and_split_data, fetch_stock_data, preprocess_data

# Directory where models are stored
MODEL_DIR = "models/lstm_models"

# Function to check if a model exists
def check_model_exists(ticker):
    """
    Check if the model and scaler exist for the given ticker.

    Parameters:
    - ticker: The stock ticker symbol.

    Returns:
    - bool: True if both model and scaler exist, False otherwise.
    """
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm_model.pt")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    return os.path.exists(model_path) and os.path.exists(scaler_path)

# Function to train a new model if it doesn't exist
def train_or_load_model(ticker, start_date=None, end_date=None):
    """
    Train a new model if it doesn't exist, or load an existing one.

    Parameters:
    - ticker: The stock ticker symbol.
    - start_date: The start date for fetching data.
    - end_date: The end date for fetching data.

    Returns:
    - model: The trained or loaded LSTM model.
    - scaler: The MinMaxScaler used for scaling.
    """
    if check_model_exists(ticker):
        print(f"Model for {ticker} already exists. Loading model...")
        model, scaler = load_model_and_scaler(ticker)
    else:
        print(f"Model for {ticker} does not exist. Training new model...")
        train_data, val_data, test_data, scaler = fetch_process_and_split_data(ticker, start_date, end_date)
        
        num_timesteps = 60

        X_train = []
        y_train = []

        for i in range(num_timesteps, len(train_data)):
            X_train.append(train_data.iloc[i-num_timesteps:i].drop(columns=["close"]).values)
            y_train.append(train_data.iloc[i]["close"])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_val = []
        y_val = []

        for i in range(num_timesteps, len(val_data)):
            X_val.append(val_data.iloc[i-num_timesteps:i].drop(columns=["close"]).values)
            y_val.append(val_data.iloc[i]["close"])

        X_val = np.array(X_val)
        y_val = np.array(y_val)

        model = train_lstm_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            ticker=ticker
        )
        save_model_and_scaler(model, scaler, ticker)
        model, scaler = load_model_and_scaler(ticker)
        
    return model, scaler

# Function to make predictions using the loaded or newly trained model
def predict_stock_price(ticker):
    try:
        # Assuming you have already defined start_date and end_date
        start_date = "2020-01-01"
        end_date = "2023-01-01"

        # Load model and scaler
        model, scaler = train_or_load_model(ticker, start_date, end_date)

        # Fetch and process the latest data
        train_data, val_data, test_data, scaler = fetch_process_and_split_data(ticker, start_date, end_date)
        processed_data = test_data
        # Ensure the 'close' feature is included during scaling
        scaled_data = scaler.transform(processed_data)
        
        # Drop 'close' after scaling, before prediction
        scaled_data = pd.DataFrame(scaled_data, columns=processed_data.columns)
        X_latest = scaled_data.drop(columns=["close"]).values

        # Convert to tensor
        X_latest = torch.tensor(X_latest[-60:], dtype=torch.float32).unsqueeze(0)  # Assuming you need the last 60 timesteps

        # Make the prediction
        with torch.no_grad():
            predicted_value_scaled = model(X_latest).item()

        # Reconstruct the final value (optional, depending on how you set up the scaler)
        reconstructed_data = np.array([0] * (scaled_data.shape[1] - 1) + [predicted_value_scaled]).reshape(1, -1)
        predicted_value = scaler.inverse_transform(reconstructed_data)[0][-1]

        return predicted_value
    
    except Exception as e:
        print(f"Error predicting stock price for {ticker}: {str(e)}")
        return None
