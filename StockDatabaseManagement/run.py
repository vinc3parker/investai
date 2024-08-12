import sys
import os
import mysql.connector
import pandas as pd
from db.connection import connect_to_database
from db.utilities import create_table, get_all_tickers, drop_all_tables
from services.stock_data_services import get_or_fetch_ticker_data

# Add the parent directory of 'services' to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # Connect to the database
    connection = connect_to_database()

    if connection:
        try:
            tickers = get_all_tickers(connection)
            # Step 4: Fetch and store data for each ticker in the stockTickers_sp500 table
            for ticker in tickers:
                print(f"Processing {ticker}...")
                get_or_fetch_ticker_data(ticker)
            
            print("Database population complete.")
            
        finally:
            # Close the connection
            connection.close()
            print("Database connection closed.")
    else:
        print("Failed to connect to the database.")

if __name__ == "__main__":
    main()
