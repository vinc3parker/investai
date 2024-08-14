import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import requests

def load_and_prepare_data(tickers):
    """Load and prepare data from multiple tickers."""
    all_data = []
    for ticker in tickers:
        data = pd.read_csv(f"data/processed/{ticker}.csv", parse_dates=['date'])
        data['ticker'] = ticker
        all_data.append(data)
    
    combined_data = pd.concat(all_data)
    combined_data = combined_data.sort_values(by=['ticker', 'date'])
    
    return combined_data

def preprocess_data(data):
    """Preprocess the data, scaling only the `close` values."""
    scalers = {}
    scaled_data = pd.DataFrame()

    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker]
        scaler = MinMaxScaler(feature_range=(0, 1))
        ticker_data[['close']] = scaler.fit_transform(ticker_data[['close']])
        scalers[ticker] = scaler
        scaled_data = pd.concat([scaled_data, ticker_data])

    return scaled_data, scalers

def create_lstm_model(input_shape):
    """Create a generalized LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(model, data, epochs=50, batch_size=32):
    """Train the LSTM model using data from multiple tickers."""
    x_train, y_train = [], []

    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker]['close'].values
        for i in range(60, len(ticker_data)):
            x_train.append(ticker_data[i-60:i])
            y_train.append(ticker_data[i])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    
    return model

def main(tickers):
    # Load and preprocess data
    combined_data = load_and_prepare_data(tickers)
    scaled_data, scalers = preprocess_data(combined_data)
    
    # Create and train the model
    model = create_lstm_model((60, 1))  # 60 time steps, 1 feature (close price)
    model = train_model(model, scaled_data)
    
    # Save the model and scalers
    model.save("models/generalized_lstm.h5")
    for ticker, scaler in scalers.items():
        joblib.dump(scaler, f"models/{ticker}_scaler.pkl")
    
    print("Model training complete and saved.")

if __name__ == "__main__":
    url = f"http://127.0.0.1:5000/tickers"
    response = requests.get(url)
    tickers = response.json()
    main(tickers)
