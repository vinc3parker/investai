from flask import Flask, request, jsonify
from db.utilities import update_stock_data  # Assuming this function exists
from services.stock_data_services import fetch_ticker_data
app = Flask(__name__)

@app.route('/update/<ticker>', methods=['POST'])
def update_data(ticker):
    """Endpoint to trigger stock data update for a specific ticker."""
    response = update_stock_data(ticker)
    return jsonify(response)

@app.route('/fetch/<ticker>', methods=['GET'])
def fetch_data(ticker):
    """Endpoint to fetch stock data for a specific ticker."""
    data = fetch_ticker_data(ticker)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
