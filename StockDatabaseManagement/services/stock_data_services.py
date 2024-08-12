import yfinance as yf
import pandas as pd
import mysql
from db.connection import connect_to_database
from db.utilities import table_exists, fetch_ticker_data, create_table, insert_data_into_table, data_up_to_date, drop_all_tables, remove_ticker_from_table, drop_table, get_all_tickers
from services.data_preparation import prepare_data_for_insert

def get_or_fetch_ticker_data(ticker_raw):
    """Ensure the ticker data is up to date, fetch if not."""
    ticker_db = f"stock_{ticker_raw}"  # The name for the MySQL table
    connection = connect_to_database()
    if not connection:
        print("Failed to connect to the database.")
        return None

    try:
        if not table_exists(connection, ticker_db):
            print(f"Table for {ticker_db} does not exist. Creating table and fetching data...")
            create_table(connection, ticker_raw)
            data = fetch_data_from_yfinance(ticker_raw)
            if data.empty:
                raise ValueError(f"Insufficient data for {ticker_raw}")
            print("Inserting Data")
            insert_data_into_table(connection, ticker_db, data)
        else:
            if not data_up_to_date(connection, ticker_db):
                print("Data is not up to date, fetching new data...")
                data = fetch_data_from_yfinance(ticker_raw)
                if data.empty:
                    raise ValueError(f"Insufficient data for {ticker_raw}")
                insert_data_into_table(connection, ticker_db, data)
            else:
                print("Data is up to date.")
                data = fetch_data(connection, ticker_db)
        return data
    except ValueError as ve:
        print(f"{ve}. Dropping table {ticker_db} and removing from stockTickers_sp500")
        drop_table(connection, ticker_db)
        remove_ticker_from_table(connection, ticker_raw)
    except mysql.connector.Error as err:
        print(f"Error with ticker {ticker_raw}: {err}")
        return None
    finally:
        connection.close()


def fetch_data_from_yfinance(ticker):
    """Fetches data from yfinance for the given ticker."""
    ticker = ticker.replace('.', '-')  # Ensure ticker is in the correct format for yfinance
    stock = yf.Ticker(ticker)
    
    try:
        # Fetch data for the last 5 years
        data = stock.history(period="5y")  # Fetch data for the last 5 years

        # Check if the data is empty
        if data.empty:
            print(f"No data found for {ticker} over the last 5 years.")
            # Return an empty DataFrame with the expected columns
            data = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        else:
            # Convert the index to date only if it's a datetime with timezone
            if data.index.dtype == 'datetime64[ns, UTC]':
                data.index = data.index.tz_convert(None).normalize()  # Remove timezone and normalize to midnight

            # Ensure that the 'Date' column is present
            data.reset_index(inplace=True)

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        # Return an empty DataFrame with the expected columns
        data = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # Prepare the data for insertion (assumed to be a separate function)
    prepared_data = prepare_data_for_insert(data)
    return prepared_data

def fetch_data(connection, ticker_db):
    print(f"Fetching existing data for {ticker_db}.")
    raw_data = fetch_ticker_data(connection, ticker_db)
    
    if raw_data:
        df = pd.DataFrame(raw_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        print(f"Data fetched successfully for {ticker_db}.")
        return df
    else:
        print(f"No data found for {ticker_db}.")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found

def get_ticker_list():
    connection = connect_to_database()
    data = get_all_tickers(connection)
    return data