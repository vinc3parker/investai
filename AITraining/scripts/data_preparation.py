import pandas as pd
import numpy as np
from ta import add_all_ta_features, add_trend_ta, add_volume_ta, add_volatility_ta, add_momentum_ta
from ta.utils import dropna
import requests
import os

def load_data(ticker):
    # Correct the path to the processed data
    file_path = f'data/processed/{ticker}.csv'  # Adjust path as needed
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    data = pd.read_csv(file_path, parse_dates=['date'])
    return data

def add_technical_indicators(data):
    """Add technical indicators to the data"""
    # Ensure the data isn't missing any points
    data = dropna(data)
    
    print("Data before adding indicators:")
    print(data.describe())  # Overview of the data
    
    # Add indicators incrementally
    try:
        data = add_trend_ta(data, "open", "high", "low", "close", "volume")
        print("Added trend indicators.")
        
        data = add_volume_ta(data, "open", "high", "low", "close", "volume")
        print("Added volume indicators.")
        
        data = add_volatility_ta(data, "open", "high", "low", "close", "volume")
        print("Added volatility indicators.")
        
        data = add_momentum_ta(data, "open", "high", "low", "close", "volume")
        print("Added momentum indicators.")
        
    except Exception as e:
        print(f"Error adding technical indicators: {e}")

    return data



def split_data(data):
    """Split the data into the training, validation and testing data sets"""
    # Sort the data by date to ensure correct order
    data = data.sort_values(by='date')

    # Find most recent date in data
    most_recent_date = data['date'].max()
    cutoff_date = most_recent_date - pd.DateOffset(years=1)

    # Use the first 4 years of data for training and validation
    training_data = data[data['date'] < cutoff_date]

    testing_data = data[data['date'] >= cutoff_date]

    return training_data, testing_data

def prepare_data_for_modeling(ticker):
    # Load the data
    data = load_data(ticker)
    # Add technical indicators
    data = add_technical_indicators(data)
    # Split the data
    training_data, testing_data = split_data(data)

    return training_data, testing_data

def save_data(training, testing, ticker):
    output_dir_train = "data/prepared_training"  # Correct the path
    output_dir_test = "data/prepared_testing"  # Correct the path

    # Ensure directories exist
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_test, exist_ok=True)

    # For training data
    file_path_train = os.path.join(output_dir_train, f"{ticker}.csv")
    training.to_csv(file_path_train, index=False)
    print(f"Training data for {ticker} saved to {file_path_train}")

    # For testing data
    file_path_test = os.path.join(output_dir_test, f"{ticker}.csv")
    testing.to_csv(file_path_test, index=False)
    print(f"Testing data for {ticker} saved to {file_path_test}")

def create_data():
    url = f"http://127.0.0.1:5000/tickers"
    response = requests.get(url)
    tickers = response.json()
    for ticker in tickers:
        training, testing = prepare_data_for_modeling(ticker)
        save_data(training, testing, ticker)


if __name__ == "__main__":
    # Prepare data, split and save
    create_data()
