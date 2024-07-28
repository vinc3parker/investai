import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables

def connect_to_database():
    """Connect to the MySQL database and return the connection."""
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Error connecting to the MySQL database: {err}")
        return None
