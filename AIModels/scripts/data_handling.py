import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from ta.utils import dropna
import os
import joblib

# Function to fetch data from StockDatabaseManagement service
def fetch_stock_data(ticker, start_date=None, end_date=None):
    url = f"http://127.0.0.1:5000/fetch/{ticker}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = pd.DataFrame(response.json())
        
        # Convert 'date' column to datetime for easy filtering
        data['date'] = pd.to_datetime(data['date'])

        # Filter data based on start_date and end_date
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data['date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data['date'] <= end_date]
        
        return data
    
    else:
        raise ValueError(f"Error fetching data for {ticker}: {response.json().get('message', 'Unknown error')}")

# Function to preprocess data
def preprocess_data(data):
    data = dropna(data)  # Drop any missing values

    # Add technical indicators
    data = add_all_ta_features(
        data, open="open", high="high", low="low", close="close", volume="volume"
    )

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.drop(columns=["date"]))
    
    # Recombine with dates for reference
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns.drop("date"))
    scaled_df["date"] = data["date"].values
    
    return scaled_df, scaler

# Function to fetch, process, and scale data
def fetch_and_process_data(ticker, start_date=None, end_date=None):
    # Fetch data
    raw_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Process and scale data
    processed_data, scaler = preprocess_data(raw_data)
    
    return processed_data, scaler

# Save processed data to a file
def save_processed_data(ticker, processed_data, scaler, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    
    data_file = os.path.join(output_dir, f"{ticker}_processed.csv")
    scaler_file = os.path.join(output_dir, f"{ticker}_scaler.pkl")
    
    processed_data.to_csv(data_file, index=False)
    joblib.dump(scaler, scaler_file)
    
    print(f"Processed data and scaler for {ticker} saved to {output_dir}")

# Main function to be used when running the script directly
if __name__ == "__main__":
    ticker = "AAPL"  # Example ticker
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    processed_data, scaler = fetch_and_process_data(ticker, start_date, end_date)
    save_processed_data(ticker, processed_data, scaler)
