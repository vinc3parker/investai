# PortfolioBot/tests/test_model_prediction.py
# Add the PortfolioBot directory to the system path
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_handler import load_model_and_scaler, predict_with_model
from utils.data_handler import fetch_data_locally, fetch_live_data_from_db

def test_predictions_for_ticker(ticker, use_generic_model=True, live_data=False):
    """
    Test model predictions for a specific ticker.
    
    Parameters:
        ticker (str): The stock ticker.
        use_generic_model (bool): Whether to use the generic model or the ticker-specific model.
        live_data (bool): Whether to use live data from StockDatabaseManagement or testing data from AITraining.
    
    Returns:
        DataFrame: A DataFrame containing the predictions.
    """
    model_type = "generic" if use_generic_model else "specific"
    
    # Load the model and scaler
    if model_type == "generic":
        model, scaler = load_model_and_scaler()
    elif model_type == "specific":
        model, scaler = load_model_and_scaler(ticker, model_type=model_type)
    
    
    # Fetch data
    if live_data:
        df = fetch_live_data_from_db(ticker)
    else:
        df = fetch_data_locally(ticker)
    
    # Make predictions
    predictions = predict_with_model(model, scaler, df)
    
    # Combine predictions with the original data for comparison
    prediction_dates = df['date'].iloc[-len(predictions):].reset_index(drop=True)
    prediction_df = pd.DataFrame({'date': prediction_dates, 'predicted_close': predictions.flatten()})
    
    print(f"Predictions for {ticker} using {'generic' if use_generic_model else 'specific'} model:")
    print(prediction_df)
    
    return prediction_df

if __name__ == "__main__":
    ticker = "AAPL"  # Replace with any ticker you want to test
    
    # Test with the generic model using testing data
    test_predictions_for_ticker(ticker, use_generic_model=True, live_data=False)
    
    # Test with the ticker-specific model using live data
    test_predictions_for_ticker(ticker, use_generic_model=False, live_data=True)
