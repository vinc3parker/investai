import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import pandas as pd
import requests
from ta import add_all_ta_features
from ta.utils import dropna

# Define the LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out

# Function to fetch stock data from StockDatabaseManagement API
def fetch_stock_data(ticker, start_date=None, end_date=None):
    url = f"http://127.0.0.1:5000/fetch/{ticker}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = pd.DataFrame(response.json())
        data['date'] = pd.to_datetime(data['date'])

        if start_date:
            data = data[data['date'] >= start_date]
        if end_date:
            data = data[data['date'] <= end_date]
        
        return data
    else:
        raise ValueError(f"Error fetching data for {ticker}: {response.json().get('message', 'Unknown error')}")

# Function to preprocess data
def preprocess_data(data, scaler=None, fit_scaler=True):
    # Apply all technical indicators and drop NaN values
    data = dropna(data)
    data = add_all_ta_features(data, open="open", high="high", low="low", close="close", volume="volume")
    
    # Select the same set of features consistently
    selected_features = [
        'open', 'high', 'low', 'close', 'volume', 
        'momentum_rsi', 'trend_macd', 'trend_macd_signal', 'volatility_atr'
    ]
    
    data = data[selected_features]
    data.dropna(inplace=True)

    # Print the head of the data before scaling
    print("Head of the data before scaling:")
    print(data.head())

    # Fit or transform the scaler based on the given data
    if fit_scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided if fit_scaler is set to False.")
        scaled_data = scaler.transform(data)

    scaled_df = pd.DataFrame(scaled_data, columns=selected_features)
    
    return scaled_df, scaler


# Function to split data into training, validation, and testing sets
def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=0.2, shuffle=False)
    
    return train_data, val_data, test_data

# Function to train the LSTM model
def train_stock_specific_model(train_data, val_data, ticker, scaler):
    X_train = train_data.drop(columns=["close"]).values
    y_train = train_data["close"].values
    X_val = val_data.drop(columns=["close"]).values
    y_val = val_data["close"].values

    # Print the head of the training data before reshaping
    print("Head of the training data:")
    print(train_data.head())

    X_train = torch.tensor(X_train, dtype=torch.float32).view(X_train.shape[0], -1, X_train.shape[1])
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32).view(X_val.shape[0], -1, X_val.shape[1])
    y_val = torch.tensor(y_val, dtype=torch.float32)

    model = StockLSTM(input_size=X_train.shape[2], hidden_size=50, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        val_output = model(X_val)
        val_loss = criterion(val_output.squeeze(), y_val).item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Validation Loss: {val_loss}")

    model_dir = "models/lstm_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{ticker}_lstm_model.pt")
    scaler_path = os.path.join(model_dir, f"{ticker}_scaler.pkl")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)  # Save the scaler

    return model_path, scaler_path

# Function to load the model and scaler
def load_model_and_scaler(ticker):
    model_dir = "models/lstm_models"
    model_path = f"{model_dir}/{ticker}_lstm_model.pt"
    scaler_path = f"{model_dir}/{ticker}_scaler.pkl"

    model = StockLSTM(input_size=8, hidden_size=50, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler

# Function to fetch, process, and scale the latest data
def fetch_and_process_latest_data(ticker, num_days=60, scaler=None):
    data = fetch_stock_data(ticker).tail(num_days)
    processed_data, _ = preprocess_data(data, scaler=scaler, fit_scaler=False)

    return processed_data



# Function to predict the next stock value using the saved model
def predict_next_value(ticker, num_days=60):
    model, scaler = load_model_and_scaler(ticker)
    processed_data = fetch_and_process_latest_data(ticker, num_days, scaler=scaler)

    # Ensure the 'close' column is excluded from the prediction input
    if 'close' in processed_data.columns:
        processed_data = processed_data.drop(columns=['close'])

    # Print the head of the data before prediction
    print("Head of the data before prediction:")
    print(processed_data.head())

    # Scale the data using the previously saved scaler
    scaled_data = scaler.transform(processed_data.values)

    # Prepare the data for prediction
    X = torch.tensor(scaled_data, dtype=torch.float32).view(1, -1, scaled_data.shape[1])

    with torch.no_grad():
        predicted_value_scaled = model(X).item()

    # Inverse the scaling for the prediction to get the actual value
    predicted_value = scaler.inverse_transform(
        np.array([[0] * (scaled_data.shape[1] - 1) + [predicted_value_scaled]])
    )[0, -1]

    return predicted_value



# Main function to handle training and prediction
def main():
    ticker = 'META'
    
    try:
        print("Predicting next value...")
        prediction = predict_next_value(ticker)
        print(f"The predicted next value for {ticker} is {prediction}")

    except FileNotFoundError as e:
        print(f"Model or scaler not found for {ticker}, training a new model.")
        
        # Fetch and preprocess data
        print("Fetching data...")
        start_date = "2020-01-01"
        end_date = "2023-01-01"
        data = fetch_stock_data(ticker, start_date, end_date)
        
        print("Processing data...")
        processed_data, scaler = preprocess_data(data)
        
        # Split the data
        print("Splitting data...")
        train_data, val_data, _ = split_data(processed_data)
        
        # Train the model
        print("Training model...")
        model_path, scaler_path = train_stock_specific_model(train_data, val_data, ticker, scaler)
        
        # Try predicting again after training
        print("Predicting next value after training...")
        prediction = predict_next_value(ticker)
        print(f"The predicted next value for {ticker} after training is {prediction}")

if __name__ == "__main__":
    main()
