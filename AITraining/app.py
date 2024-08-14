from flask import Flask, jsonify, send_file, request
import os
import joblib
from scripts.fetch_data import fetch

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_GENERIC_DIR = "models/lstm_generic_models"
MODELS_SPECIFIC_DIR = "models/lstm_multifeature_models"
DATA_DIR = "data/prepared_testing"

@app.route('/models/generic', methods=['GET'])
def download_generic_model():
    """API endpoint to download a trained model"""
    model_name = "generic_lstm_multifeature.h5"
    model_path = os.path.join(MODELS_GENERIC_DIR, model_name)
    file = fetch(model_path)
    return file

@app.route('/scaler/generic', methods=['GET'])
def download_generic_scaler():
    """API endpoint to download a trained model"""
    model_name = 'generic_scaler_multifeature.pkl'
    model_path = os.path.join(MODELS_GENERIC_DIR, model_name)
    file = fetch(model_path)
    return file
    
@app.route('/models/specific/<model_name>', methods=['GET'])
def download__specific_model(model_name):
    """API endpoint to download a trained model"""
    END_PATH = '_lstm_multifeature.h5'
    model_start = os.path.join(MODELS_SPECIFIC_DIR, model_name)
    model_path = f"{model_start}{END_PATH}"
    file = fetch(model_path)
    return file

@app.route('/scaler/specific/<model_name>', methods=['GET'])
def download_specific_scaler(model_name):
    """API endpoint to download a trained model"""
    END_PATH = '_scaler_multifeature.pkl'
    model_start = os.path.join(MODELS_SPECIFIC_DIR, model_name)
    model_path = f"{model_start}{END_PATH}"
    file = fetch(model_path)
    return file
    
# Set the base directory where your files are located
BASE_DIR = "C:/Users/Invate/OneDrive/Computing/Projects/AI & Deep Learning/Stock Bot/investai/AITraining/data/prepared_testing"

@app.route('/data/<ticker>', methods=['GET'])
def download_data(ticker):
    """API endpoint to download a CSV file of the prepared data."""
    # Construct the absolute path to the file
    data_file = os.path.join(BASE_DIR, f"{ticker}.csv")
    
    # Print the file path for debugging
    print(f"Trying to access file at: {data_file}")
    
    if os.path.exists(data_file):
        try:
            return send_file(data_file, as_attachment=True)
        except Exception as e:
            print(f"Error sending file: {str(e)}")
            return jsonify({"error": f"Error sending file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Data not found", "path": data_file}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)