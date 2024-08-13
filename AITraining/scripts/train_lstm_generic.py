import os
import pandas as pd
import numpy as np
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import requests

def load_and_aggregate_data(tickers):
    """Load and aggregate data from multiple tickers into a single DataFrame."""
    all_data = []
    for ticker in tickers:
        file_path = f"data/prepared_training/{ticker}.csv"
        data = pd.read_csv(file_path, parse_dates=['date'])
        data['ticker'] = ticker
        all_data.append(data)
    
    combined_data = pd.concat(all_data)
    combined_data = combined_data.sort_values(by=['date', 'ticker'])  # Ensure data is ordered correctly
    
    return combined_data

def split_data(data):
    """Split the data into training and validation sets."""
    X = data.drop(columns=['date', 'ticker']).values
    y = data['close'].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    return X_train, X_val, y_train, y_val

def preprocess_data(X_train, X_val):
    """Preprocess the data, scaling all relevant features."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    return X_train_scaled, X_val_scaled, scaler

def create_lstm_model(input_shape):
    """Create a more complex LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Still predicting the close price
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def reshape_for_lstm(X):
    """Reshape input data for LSTM model."""
    X_reshaped = []
    for i in range(60, len(X)):
        X_reshaped.append(X[i-60:i].astype(np.float32))  # Convert to float32 to save memory
    return np.array(X_reshaped, dtype=np.float32)  # Convert entire array to float32


def batch_train_model(model, X_train, y_train, batch_size=32, epochs=50):
    """Train the model using batch processing."""
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            model.train_on_batch(X_batch, y_batch)
        print(f"Epoch {epoch+1}/{epochs} completed.")

    return model


def evaluate_model(model, X_val, y_val):
    """Evaluate the model on the validation set."""
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f"Root Mean Squared Error on validation set: {rmse}")
    return rmse

def save_model(model, scaler):
    """Save the trained model and scaler."""
    model_dir = "models/lstm_generic_models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "generic_lstm_multifeature.h5")
    scaler_path = os.path.join(model_dir, "generic_scaler_multifeature.pkl")
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Generic model and scaler saved to {model_dir}.")

def main(tickers):
    # Load and aggregate data from multiple tickers
    combined_data = load_and_aggregate_data(tickers)
    
    # Split the aggregated data
    X_train, X_val, y_train, y_val = split_data(combined_data)
    
    # Preprocess the data
    X_train_scaled, X_val_scaled, scaler = preprocess_data(X_train, X_val)
    
    # Reshape data for LSTM
    X_train_lstm = reshape_for_lstm(X_train_scaled)
    X_val_lstm = reshape_for_lstm(X_val_scaled)
    
    # Adjust y_train and y_val for the LSTM reshaping
    y_train_lstm = y_train[60:]
    y_val_lstm = y_val[60:]
    
    # Create and train the model
    model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
    model = batch_train_model(model, X_train_lstm, y_train_lstm)
    
    # Evaluate the model
    evaluate_model(model, X_val_lstm, y_val_lstm)
    
    # Save the model and scaler
    save_model(model, scaler)

if __name__ == "__main__":
    url = f"http://127.0.0.1:5000/tickers"
    response = requests.get(url)
    tickers = response.json()
    main(tickers)
