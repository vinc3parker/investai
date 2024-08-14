import pandas as pd
import requests

def fetch_latest_data(ticker, source='StockDatabase'):
    """
    Fetch the latest 60 data points for the given ticker.

    Args:
        ticker (str): The stock ticker.
        source (str): The data source ('StockDatabase' or 'AITraining').

    Returns:
        DataFrame: The latest 60 data points.
    """
    if source == 'StockDatabase':
        url = f"http://127.0.0.1:5000/fetch/{ticker}"
    elif source == 'AITraining':
        url = f"http://127.0.0.1:5001/data/{ticker}"
    else:
        raise ValueError("Invalid source specified")

    response = requests.get(url)
    if response.status_code == 200:
        data = pd.DataFrame(response.json())
        return data.tail(60)  # Only take the last 60 data points
    else:
        raise ValueError(f"Error fetching data from {source} for {ticker}: {response.json().get('error', 'Unknown error')}")
