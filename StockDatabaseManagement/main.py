from services.stock_data_services import get_or_fetch_ticker_data

if __name__ == "__main__":
    ticker = "AAPL"  # Example ticker
    data = get_or_fetch_ticker_data(ticker)
    print(data)
