from flask import Flask, jsonify, request
from db.utilities import connect_to_database
from services.stock_data_services import fetch_ticket_data
import json

app = Flask(__name__)

@app.route('/api/stockdata', methods=['GET'])
def get_stock_data():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error': 'Ticker is required'}), 400
    
    data = fetch_ticket_data(ticker)
    if data is None or data.empty:
        return jsonify({'error': 'No data found for ticker'}), 404

    # Convert data to JSON
    result = data.to_json(orient='records')
    parsed = json.loads(result)
    return jsonify(parsed)

if __name__ == '__main__':
    app.run(debug=True)