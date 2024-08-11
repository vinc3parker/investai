from flask import Flask, request, jsonify
from flask_cors import CORS
from services.stock_data_services import get_or_fetch_ticker_data, get_all_tickers


app = Flask(__name__)
CORS(app)

@app.route('/fetch/<ticker>', methods=['GET'])
def fetch_data(ticker):
    """Endpoint to fetch stock data for a specific ticker."""
    data = get_or_fetch_ticker_data(ticker)
    if not data.empty:
        return jsonify(data.to_dict(orient='records'))
    else:
        return jsonify({'messge': 'Error when looking for the specified ticker data'}), 404

@app.route('/tickers', methods=['GET'])
def tickers():
    tickers = get_all_tickers()
    if tickers is not None:
        return jsonify(tickers)
    else:
        return jsonify({"error": "Failed to fetch tickers"}), 500 

if __name__ == '__main__':
    app.run(debug=True, port=5000)
