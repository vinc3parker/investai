import yfinance as yf
import pandas as pd
from db.connection import connect_to_database
from db.utilities import table_exists, fetch_ticker_data, create_table, insert_data_into_table, data_up_to_date
from services.data_preparation import prepare_data_for_insert

def get_or_fetch_ticker_data(ticker):
    """Ensure the ticker data is up to date, fetch if not."""
    connection = connect_to_database()
    if not connection:
        print("Failed to connect to the database.")
        return None

    try:
        if not table_exists(connection, ticker):
            print(f"Table for {ticker} does not exist. Creating table and fetching data...")
            create_table(connection, ticker)
            data = fetch_data_from_yfinance(ticker)
            print("Inserting Data")
            insert_data_into_table(connection, ticker, data)
        else:
            if not data_up_to_date(connection, ticker):
                print("Data is not up to date, fetching new data...")
                data = fetch_data_from_yfinance(ticker)
                insert_data_into_table(connection, ticker, data)
            else:
                print("Data is up to date.")
                data = fetch_data(connection, ticker)
        return data
    except mysql.connector.Error as err:
        print(f"Error with ticker {ticker}: {err}")
        return None
    finally:
        connection.close()


def fetch_data_from_yfinance(ticker):
    """Fetches data from yfinance for the given ticker."""
    ticker = ticker.replace('.', '-')
    stock = yf.Ticker(ticker)
    # Here you might customize the period and interval
    data = stock.history(period="1mo")  # Fetch data for the last month

    # Convert the index to date only if it's a datetime with timezone
    if data.index.dtype == 'datetime64[ns, UTC]':
        data.index = data.index.tz_convert(None).normalize()  # Remove timezone and normalize to midnight

    prepared_data = prepare_data_for_insert(data)
    return prepared_data

def fetch_data(connection, ticker):
    print(f"Fetching existing data for {ticker}.")
    raw_data = fetch_ticker_data(connection, ticker)
    
    if raw_data:
        df = pd.DataFrame(raw_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        print(f"Data fetched successfully for {ticker}.")
        return df
    else:
        print(f"No data found for {ticker}.")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found


def get_all_tickers():
    """Fetch all ticker labels from the StockTickers table."""
    connection = connect_to_database()
    if not connection:
        print("Failed to connect to the database.")
        return None

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT ticker FROM StockTickers")
        tickers = cursor.fetchall()
        cursor.close()
        
        if tickers:
            print("Fetched all tickers successfully.")
            return [ticker['ticker'] for ticker in tickers]
        else:
            print("No tickers found.")
            return []

    finally:
        connection.close()

def fetch_sp500_tickers():
    """Fetch the list of S&P 500 tickers from an online source."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_table = tables[0]
    sp500_tickers = sp500_table['Symbol'].tolist()
    return sp500_tickers

def populate_database_with_sp500():
    """Fetch and store data for all S&P 500 tickers."""
    sp500_tickers = fetch_sp500_tickers()
    for ticker in sp500_tickers:
        print(f"Processing {ticker}...")
        get_or_fetch_ticker_data(ticker)
    print("Database population complete.")