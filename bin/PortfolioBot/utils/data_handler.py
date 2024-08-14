# PortfolioBot/utils/data_handler.py
import os
import pandas as pd
import requests

def fetch_live_data_from_db(ticker, connection, start_date, end_date):
    """
    Fetch live data from the StockDatabaseManagement MySQL database.
    
    Parameters:
        ticker (str): The stock ticker.
        connection: The MySQL database connection.
        start_date (str): The start date for the data.
        end_date (str): The end date for the data.
    
    Returns:
        DataFrame: A DataFrame containing the stock data.
    """
    url = f"http://127.0.0.1:5000/fetch/{ticker}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        raise ValueError(f"Error fetching live data from StockDatabaseManagement for {ticker}: {response.json().get('message', 'Unknown error')}")

DATA_DIR = "data/prepared_testing"

def fetch_data_locally(ticker):
    """Fetch prepared data for a specific ticker from the local directory."""
    data_file = os.path.join(DATA_DIR, f"{ticker}.csv")
    
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, parse_dates=['date'])
        return df
    else:
        raise ValueError(f"Data not found for {ticker} at path {data_file}")
