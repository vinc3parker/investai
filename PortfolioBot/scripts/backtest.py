import os
import sys
# Add the PortfolioBot directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_handler import fetch_data, load_data, fetch_model, load_model
from strategies.momentum_strategy import apply_momentum_strategy
from strategies.mean_reversion_strategy import apply_mean_reversion_strategy

def backtest_strategy(ticker, model_name, strategy_function):
    """Backtest a given strategy on a specific ticker's data."""

    # Fetch and load the data
    data_path = fetch_data(ticker)
    df = load_data(data_path)

    if df is not None:
        # Fetch and load the model
        model_path = fetch_model(model_name)
        model = load_model(model_path)

        # Simulate trades and calculate performance using the provided strategy
        if model is not None:
            # Here, you could incorporate model predictions as part of the strategy
            # Simulate trades and calculate performance
            trade_log, final_value, total_return = simulate_trades(strategy_function, df)

            # Return the results for analysis
            return trade_log, final_value, total_return
    return None, None, None

if __name__ == "__main__":
    # Example backtest for a single ticker using momentum strategy
    ticker = 'AAPL'  # Replace with any ticker you want to test
    model_name = 'generic_lstm_multifeature.h5'
    trade_log, final_value, total_return = backtest_strategy(ticker, model_name, apply_momentum_strategy)

    if trade_log:
        print(f"Backtest complete for {ticker}. Final Value: ${final_value:.2f}, Total Return: {total_return:.2f}%")
        for trade in trade_log:
            print(trade)
    else:
        print(f"Backtest failed for {ticker}.")
