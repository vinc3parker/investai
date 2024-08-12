import mysql.connector
import datetime
import pandas_market_calendars as mcal
from services.data_preparation import prepare_data_for_insert

# Create ticket table
def create_ticker_table(connection):
    cursor = connection.cursor()
    create_statement = """
    CREATE TABLE IF NOT EXISTS StockTickers (
        id INT AUTO_INCREMENT PRIMARY KEY,
        ticker VARCHAR(10) UNIQUE NOT NULL,
        create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(create_statement)
    connection.commit()
    cursor.close()
    print("StockTikers table created successfully.")

# Checks if table exists
def table_exists(connection, table_name):
    cursor = connection.cursor()
    cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
    result = cursor.fetchone()
    cursor.close()
    return result is not None

# Fetches ticker data from relevant ticket table
def fetch_ticker_data(connection, ticker):
    cursor = connection.cursor()
    try:
        query = f"SELECT * FROM `{ticker}`"
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except mysql.connector.Error as err:
        print(f"Error fetching data from the database for ticker {ticker}: {err}")
        return []
    finally:
        cursor.close()

# Creates a table for the a ticker
def create_table(connection, ticker):
    cursor = connection.cursor()
    table_name = f"stock_{ticker}"
    create_statement = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        date DATE,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume BIGINT
    );
    """
    cursor.execute(create_statement)
    connection.commit()
    cursor.close()
    print(f"Table {table_name} created successfully.")

# Adding data to a table
def insert_data_into_table(connection, table_name, data):
    """Insert data from a pandas DataFrame into the specified table."""
    cursor = connection.cursor()
    try:
        for index, row in data.iterrows():
            sql = f"INSERT INTO {table_name} (date, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s)"
            values = (row['date'], row['open'], row['high'], row['low'], row['close'], row['volume'])
            cursor.execute(sql, values)
        connection.commit()
        print("Data inserted successfully.")
    except mysql.connector.Error as e:
        print(f"Error inserting data: {e}")
        connection.rollback()
    finally:
        cursor.close()

# Checks if the date is a trading day
def is_trading_day(date):
    # Get the NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=date, end_date=date)
    return not schedule.empty

# Finds the last day the market was open
def get_last_trading_day():
    today = datetime.date.today()
    one_week_ago = today - datetime.timedelta(days=7)
    
    # Get the NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.valid_days(start_date=one_week_ago, end_date=today)
    
    if not schedule.empty:
        # Return the last valid trading day in the schedule
        return schedule[-1].date()
    return None

# Checks if ticker data is up to date
def data_up_to_date(connection, table_name):
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT MAX(date) FROM {table_name}")
        result = cursor.fetchone()
        if result and result[0]:
            last_date_in_db = result[0]
            last_trading_day = get_last_trading_day()
            
            if last_trading_day and last_date_in_db >= last_trading_day:
                return True
            else:
                return False
        return False
    except mysql.connector.Error as err:
        print(f"Error checking if data is up to date in {table_name}: {err}")
        return False
    finally:
        cursor.close()

def create_ticker_table(connection):
    cursor = connection.cursor()
    create_statement = """
    CREATE TABLE IF NOT EXISTS StockTickers (
        id INT AUTO_INCREMENT PRIMARY KEY,
        ticker VARCHAR(50) UNIQUE NOT NULL,  # Increased from VARCHAR(10) to VARCHAR(50)
        create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(create_statement)
    connection.commit()
    cursor.close()
    print("StockTickers table created successfully.")


def get_all_tickers(connection):
    """Fetch all ticker symbols from the stockTickers_sp500 table."""
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT ticker FROM stockTickers_sp500")
        tickers = cursor.fetchall()
        if tickers:
            return [ticker['ticker'] for ticker in tickers]
        else:
            print("No tickers found in the stockTickers_sp500 table.")
            return []
    except mysql.connector.Error as err:
        print(f"Error fetching tickers from the stockTickers_sp500 table: {err}")
        return []
    finally:
        cursor.close()

def update_ticker_table(connection):
    exclusion_list = ['StockTickers', 'run']
    """Update the StockTickers table with all existing table names in the database, excluding those in exclusion_list."""
    cursor = connection.cursor()
    try:
        # Ensure the StockTickers table exists
        create_ticker_table(connection)

        # Get all table names in the database except the StockTickers table
        cursor.execute("SHOW TABLES")
        all_tables = cursor.fetchall()

        # Convert list of tuples to a flat list of table names
        all_tables = [table[0] for table in all_tables if table[0] not in exclusion_list]

        for table_name in all_tables:
            # Check if the table name is already in the StockTickers table
            cursor.execute("SELECT ticker FROM StockTickers WHERE ticker = %s", (table_name,))
            result = cursor.fetchone()

            if not result:
                # If the ticker is not found in StockTickers, insert it
                cursor.execute("INSERT INTO StockTickers (ticker) VALUES (%s)", (table_name,))
                connection.commit()
                print(f"Ticker {table_name} added to the StockTickers table.")

        print("Ticker table update complete.")
        
    except mysql.connector.Error as err:
        print(f"Error updating the ticker table: {err}")
    finally:
        cursor.close()

def drop_all_tables(connection):
    """Drops all tables in the database."""
    cursor = connection.cursor()
    try:
        # Fetch all table names
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()

        # Drop each table
        for (table_name,) in tables:
            print(f"Dropping table {table_name}...")
            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
        
        connection.commit()
        print("All tables dropped successfully.")
    except mysql.connector.Error as e:
        print(f"Error dropping tables: {e}")
    finally:
        cursor.close()

def drop_table(connection, table_name):
    cursor = connection.cursor()
    try:
        cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
        connection.commit()
        print(f"Table {table_name} dropped successfullly.")
    except mysql.connector.Error as e:
        print(f"Error dropping table {table_name}: {e}")
    finally:
        cursor.close()

def remove_ticker_from_table(connection, ticker):
    """Removes the ticker entry from stockTickers_sp500 table for now"""
    cursor = connection.cursor()
    try:
        cursor.execute("DELETE FROM stockTickers_sp500 WHERE ticker = %s", (ticker,))
        connection.commit()
        print(f"Ticker {ticker} remove from stockTickers_sp500.")
    except mysql.connector.Error as e:
        print(f"Error removing ticker {ticker}: {e}")
    finally: 
        cursor.close()

def create_stock_tickers_table(connection):
    """Creates the stockTickers_sp500 table."""
    cursor = connection.cursor()
    try:
        create_statement = """
        CREATE TABLE IF NOT EXISTS `stockTickers_sp500` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ticker VARCHAR(10) UNIQUE NOT NULL,
            full_name VARCHAR(255) UNIQUE NOT NULL
        );
        """
        cursor.execute(create_statement)
        connection.commit()
        print("stockTickers_sp500 table created successfully.")
    except mysql.connector.Error as e:
        print(f"Error creating stockTickers_sp500 table: {e}")
    finally:
        cursor.close()