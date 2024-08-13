import os
import subprocess
from dotenv import load_dotenv
from datetime import datetime

# Configuration
DB_USER=os.getenv('DB_HOST')
DB_PASSWORD=os.getenv('DB_PASSWORD')
DB_NAME=os.getenv('DB_NAME')
BACKUP_DIR = "dbFiles"

# Create backup directory if it doesn't exist
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

# Create a filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_filename = f"{DB_NAME}_backup_{timestamp}.sql"
backup_filepath = os.path.join(BACKUP_DIR, backup_filename)

# Command to dump the MySQL database
dump_command = f"mysqldump -u {DB_USER} -p{DB_PASSWORD} {DB_NAME} > {backup_filepath}"

# Run the command
try:
    subprocess.run(dump_command, shell=True, check=True)
    print(f"Backup successful! File saved as: {backup_filepath}")
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    print("Backup failed.")
