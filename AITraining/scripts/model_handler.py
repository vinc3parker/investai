import numpy as np
from keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
import requests
import os
import joblib

# Ensure this path is correctly set based on where the models are saved.
GENERIC_MODEL_PATH = 'AITraining/models/lstm_generic_models/generic_lstm_multifeature.h5'
GENERIC_SCALER_PATH = 'AITraining/models/lstm_generic_models/generic_scaler_multifeature.pkl'

def predict_with_generic_model(data):
    """
    Predict the next value using the generic LSTM model.

    Args:
        data (DataFrame): The last 60 data points of the stock.

    Returns:
        float: The predicted value.
    """
    # Load the model and scaler
    model = load_model(GENERIC_MODEL_PATH)
    scaler = joblib.load(GENERIC_SCALER_PATH)

    # Preprocess the data
    data_scaled = scaler.transform(data)

    # Reshape for LSTM model (1 sample, 60 timesteps, number of features)
    data_reshaped = np.array([data_scaled])

    # Make prediction
    prediction_scaled = model.predict(data_reshaped)
    prediction = scaler.inverse_transform(prediction_scaled)[0][0]

    return prediction
