import pandas as pd
import os
import requests

# Base directory for the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
# Paths to processed data directory
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed')
print(PROCESSED_DATA_PATH)

def fetch_data(ticker):
    """Fetches data for a specific ticker from the StockDatabaseManagement service."""
    url = f"http://127.0.0.1:5000/fetch/{ticker}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        if df.empty:
            print(f"No data available for {ticker}.")
            return None
        else:
            return df
    else:
        print(f"Failed to fetch data for {ticker}. HTTP Status code: {response.status_code}")
        return None

def process_data(df):
    """Processes the raw data."""
    if df is None or df.empty:
        return None
    
    # Convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Optional: Add any processing steps here, like handling missing values, etc.
    
    return df

def save_data(df, ticker):
    """Saves the processed data to a CSV file."""
    if df is None or df.empty:
        print(f"No data to save for {ticker}.")
        return
    
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(file_path, index=False)
    print(f"Data for {ticker} saved to {file_path}.")

def process_and_save_ticker_data(ticker):
    """Fetches, processes, and saves data for a specific ticker."""
    print(f"Processing data for {ticker}...")
    
    # Fetch data
    raw_data = fetch_data(ticker)
    
    # Process data
    processed_data = process_data(raw_data)
    
    # Save data
    save_data(processed_data, ticker)

def process_all_tickers():
    """Processes and saves data for all tickers."""
    url = f"http://127.0.0.1:5000/tickers"
    response = requests.get(url)
    tickers = response.json()
    for ticker in tickers:
        process_and_save_ticker_data(ticker)

if __name__ == "__main__":
    # Process and save data for all tickers
    process_all_tickers()