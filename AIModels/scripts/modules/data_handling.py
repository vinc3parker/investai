import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from ta.utils import dropna

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
def preprocess_data(data):
    """
    Preprocess the stock data by adding technical indicators, dropping NaNs, and scaling.

    Parameters:
    - data: The raw stock data.

    Returns:
    - scaled_df: The preprocessed and scaled data.
    - scaler: The scaler used for scaling the data.
    """
    # Drop NaN values
    data = dropna(data)

    # Add technical indicators
    data = add_all_ta_features(
        data, open="open", high="high", low="low", close="close", volume="volume"
    )
    
    # Select specific features for model training
    selected_features = [
        'open', 'high', 'low', 'close', 'volume', 
        'momentum_rsi', 'trend_macd', 'trend_macd_signal', 'volatility_atr'
    ]
    
    data = data[selected_features]
    data.dropna(inplace=True)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=selected_features)
    print(scaled_df)

    return scaled_df, scaler

# Function to split data into training, validation, and testing sets
def split_data(data, test_size=0.2, val_size=0.2):
    """
    Split the data into training, validation, and testing sets.

    Parameters:
    - data: The preprocessed data.
    - test_size: The proportion of data to be used for testing.
    - val_size: The proportion of data to be used for validation from the training set.

    Returns:
    - train_data: Training data.
    - val_data: Validation data.
    - test_data: Testing data.
    """
    total_size = len(data)
    test_split = int(total_size * (1 - test_size))
    val_split = int(test_split * (1 - val_size))

    train_data = data[:val_split]
    val_data = data[val_split:test_split]
    test_data = data[test_split:]

    return train_data, val_data, test_data

# Function to fetch, preprocess, and split the data
def fetch_process_and_split_data(ticker, start_date=None, end_date=None):
    """
    Fetch, preprocess, and split the data for a given stock ticker.

    Parameters:
    - ticker: The stock ticker symbol.
    - start_date: Start date for fetching data.
    - end_date: End date for fetching data.

    Returns:
    - train_data: Training data.
    - val_data: Validation data.
    - test_data: Testing data.
    - scaler: The scaler used for scaling the data.
    """
    data = fetch_stock_data(ticker, start_date, end_date)
    processed_data, scaler = preprocess_data(data)
    train_data, val_data, test_data = split_data(processed_data)
    
    return train_data, val_data, test_data, scaler
