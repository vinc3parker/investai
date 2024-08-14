from flask import Flask, jsonify, request
import os
import torch
from scripts.train_models import (
    fetch_stock_data,
    preprocess_data,
    split_data,
    train_stock_specific_model,
    predict_next_value,
    load_model_and_scaler,
)

app = Flask(__name__)

MODEL_DIR = "models/lstm_models"

@app.route('/predict/<ticker>', methods=['GET'])
def predict_stock_value(ticker):
    try:
        model_path = os.path.join(MODEL_DIR, f"{ticker.lower()}_lstm_model.pt")
        scaler_path = os.path.join(MODEL_DIR, f"{ticker.lower()}_scaler.pkl")
        
        # Check if the model exists
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("Fetching data...")
            data = fetch_stock_data(ticker)

            print("Processing data...")
            processed_data, scaler = preprocess_data(data)

            print("Splitting data...")
            train_data, val_data, test_data = split_data(processed_data)

            print("Training model...")
            model_path = train_stock_specific_model(train_data, val_data, ticker)

        print("Predicting next value...")
        prediction = predict_next_value(ticker)

        print(f"The predicted next value for {ticker} is {prediction}")
        return jsonify({"ticker": ticker, "predicted_value": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
