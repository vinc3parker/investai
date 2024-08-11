import sys
import os

# Add the parent directory of 'services' to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import connect_to_database
from db.utilities import get_all_tickers

def remove_stockticker_entry(connection):
    """Removes the 'StockTickers' entry from the StockTickers table."""
    cursor = connection.cursor()
    try:
        cursor.execute("DELETE FROM StockTickers WHERE ticker = 'StockTickers'")
        connection.commit()
        print("Removed 'StockTickers' entry from the StockTickers table.")
    except mysql.connector.Error as err:
        print(f"Error removing 'StockTickers' entry: {err}")
    finally:
        cursor.close()

def main():
    # Connect to the database
    connection = connect_to_database()

    if connection:
        # Remove 'StockTickers' entry from the StockTickers table
        remove_stockticker_entry(connection)

        # Retrieve and print all tickers from the StockTickers table
        tickers = get_all_tickers(connection)
        if tickers:
            print("Tickers currently stored in the StockTickers table:")
            for ticker in tickers:
                print(ticker)
        else:
            print("No tickers found in the StockTickers table.")

        # Close the connection
        connection.close()
    else:
        print("Failed to connect to the database.")

if __name__ == "__main__":
    main()
