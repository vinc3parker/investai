import sys
import os
import mysql.connector
import pandas as pd
import requests
from bs4 import BeautifulSoup
from db.connection import connect_to_database
from db.utilities import create_table, get_all_tickers, drop_all_tables
from services.stock_data_services import get_or_fetch_ticker_data

# Add the parent directory of 'services' to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fetch_sp500_tickers():
    """Fetch the S&P 500 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})

    tickers = []
    for row in table.find_all('tr')[1:]:
        ticker = row.find_all('td')[0].text.strip()
        tickers.append(ticker)

    return tickers

def create_sp500_table(connection):
    """Create the stockTickers_sp500 table if it doesn't exist."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS stockTickers_sp500 (
        id INT AUTO_INCREMENT PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL
    );
    """
    create_table(connection, create_table_query)
    print("Table stockTickers_sp500 created or already exists.")

def populate_sp500_table(connection, tickers):
    """Populate the stockTickers_sp500 table with S&P 500 tickers."""
    cursor = connection.cursor()
    for ticker in tickers:
        cursor.execute("INSERT IGNORE INTO stockTickers_sp500 (ticker) VALUES (%s);", (ticker,))
    connection.commit()
    print("S&P 500 tickers have been added to the stockTickers_sp500 table.")

def main():
    # Connect to the database
    connection = connect_to_database()

    if connection:
        try:
            # Step 1: Fetch the S&P 500 tickers from Wikipedia
            tickers = fetch_sp500_tickers()
            
            # Step 2: Create the stockTickers_sp500 table if it doesn't exist
            create_sp500_table(connection)
            
            # Step 3: Populate the table with the fetched tickers
            populate_sp500_table(connection, tickers)
            
            # Step 4: Fetch and store data for each ticker in the stockTickers_sp500 table
            tickers = get_all_tickers(connection, table_name="stockTickers_sp500")
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
