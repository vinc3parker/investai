from flask import Flask, jsonify, request
import os
import torch
from scripts.model_handling import predict_stock_price

app = Flask(__name__)

MODEL_DIR = "models/lstm_models"

@app.route('/predict/<ticker>', methods=['GET'])
def predict_stock_value(ticker):
    try:
        prediction = predict_stock_price(ticker)
        return jsonify({"ticker": ticker, "predicted_value": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
