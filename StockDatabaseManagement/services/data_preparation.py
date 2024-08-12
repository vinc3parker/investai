import pandas as pd

def prepare_data_for_insert(data):
    """
    Adjusts yfinance data to fit into a MySQL database by normalizing date formats
    and filtering columns while keeping the data in a DataFrame format.

    Args:
        data (DataFrame): The stock data fetched from yfinance.

    Returns:
        DataFrame: Data formatted and ready for database operations.
        If an error occurs, returns an empty DataFrame.
    """
    try:
        # Reset the index to make 'Date' a column and remove timezone (normalize to midnight)
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.tz_localize(None).dt.normalize()

        # Select only the necessary columns
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Optionally, rename columns to match your MySQL table's column names if they are different
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        return data
    except Exception as e:
        print(f"Error in preparing data for insert: {e}")
        # Return an empty DataFrame with the expected columns if there's an error
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
