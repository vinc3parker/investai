import mysql.connector
import datetime
import pandas_market_calendars as mcal

# Checks if table exists
def table_exists(connection, table_name):
    cursor = connection.cursor()
    cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
    result = cursor.fetchone()
    cursor.close()
    return result is not None

# Fetches ticker data from relevant ticket table
def fetch_ticker_data(connection, table_name):
    cursor = connection.cursor()
    query = f"SELECT * FROM {table_name};"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    # Convert the result into a more manageable format if necessary
    return data

# Creates a table for the a ticker
def create_table(connection, table_name):
    cursor = connection.cursor()
    create_statement = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
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
    cursor = connection.cursor()
    for record in data:
        query = f"""
        INSERT INTO {table_name} (date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        cursor.execute(query, record)  # record should be a tuple (date, open, high, low, close, volume)
    connection.commit()
    cursor.close()
    print(f"Data inserted into {table_name} successfully.")

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
