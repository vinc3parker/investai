from db.connection import connect_to_database
from db.utilities import table_exists, create_table, data_up_to_date
import yfinance as yf

def update_stock_data(ticker):
    connection = connect_to_database()
    if not connection:
        return {"error": "Database connection failed"}

    if not table_exists(connection, ticker):
        create_table(connection, ticker)
    
    # Example of updating stock data
    if not data_up_to_date(connection, ticker):
        # Fetch new data and update database
        pass  # Implementation needed

    connection.close()
    return {"success": True}

def fetch_ticker_data(ticker):
    """Fetch stock data using yfinance."""
    data = yf.Ticker(ticker).history(period="max")
    return data.to_dict()
