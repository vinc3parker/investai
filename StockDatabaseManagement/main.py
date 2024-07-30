from flask import Flask, request, jsonify
from services.stock_data_services import get_or_fetch_ticker_data

app = Flask(__name__)

@app.route('/fetch/<ticker>', methods=['GET'])
def fetch_data(ticker):
    """Endpoint to fetch stock data for a specific ticker."""
    data = get_or_fetch_ticker_data(ticker)
    if not data.empty:
        return jsonify(data.to_dict(orient='records'))
    else:
        return jsonify({'messge': 'No data found for the specified ticker'}), 404
    

if __name__ == '__main__':
    app.run(debug=True, port=5000)
