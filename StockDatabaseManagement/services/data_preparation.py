# services/stock_data_service.py
import yfinance as yf
from db.connection import connect_to_database
from db.utilities import table_exists, fetch_ticker_data, create_table, insert_data_into_table

def get_or_fetch_ticker_data(ticker):
    connection = connect_to_database()
    if not connection:
        print("Failed to connect to the database.")
        return

    try:
        if table_exists(connection, ticker):
            data = fetch_ticker_data(connection, ticker)
            if data.empty:
                print("No data found in the table. Fetching new data...")
                data = fetch_data_from_yfinance(ticker)
                insert_data_into_table(connection, ticker, data)
        else:
            print(f"Table for {ticker} does not exist. Fetching and creating...")
            create_table(connection, ticker)
            data = fetch_data_from_yfinance(ticker)
            insert_data_into_table(connection, ticker, data)

        return data
    finally:
        if connection:
            connection.close()

def fetch_data_from_yfinance(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1mo")  # Fetch data for the last month
    return data
