from scripts.data_handler import fetch_latest_data
from scripts.model_handler import predict_with_generic_model

def predict_next_value_generic(ticker):
    """
    Predict the next value of a stock using the generic LSTM model.

    Args:
        ticker (str): The stock ticker.

    Returns:
        float: The predicted value.
    """
    # Fetch the latest data
    print('fetching data')
    data = fetch_latest_data(ticker, source='StockDatabase')

    # Drop any unnecessary columns and use only the features used in training
    features = ['open', 'high', 'low', 'close', 'volume']
    data = data[features]

    # Predict the next value using the generic model
    predicted_value = predict_with_generic_model(data)

    return predicted_value

def main():
    ticker = 'AAPL' #Example for now to test
    prediction = predict_next_value_generic(ticker)
    print(f"The predicted next value for {ticker} is {prediction}")

main()