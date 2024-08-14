import os
import io
import requests
import tempfile
from keras.models import load_model
import joblib
import numpy as np

def load_model_and_scaler(ticker=None, model_type="generic"):
    if model_type == "generic":
        model_url = f"http://127.0.0.1:5001/models/generic"
        scaler_url = f"http://127.0.0.1:5001/scaler/generic"
    elif model_type == "specific" and ticker:
        model_url = f"http://127.0.0.1:5001/models/specific/{ticker}"
        scaler_url = f"http://127.0.0.1:5001/scaler/specific/{ticker}"
    else:
        raise ValueError("Invalid model type or ticker not provided for specific model.")

    model_response = requests.get(model_url)
    scaler_response = requests.get(scaler_url)

    if model_response.status_code != 200 or scaler_response.status_code != 200:
        raise FileNotFoundError(f"Model or scaler not found for {ticker if ticker else 'generic'}")

    # Save the model to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_model_file:
        temp_model_file.write(model_response.content)
        temp_model_file_path = temp_model_file.name  # Store the path

    # Now load the model after closing the file
    model = load_model(temp_model_file_path)

    # Clean up the temporary model file
    os.remove(temp_model_file_path)

    # Save the scaler to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_scaler_file:
        temp_scaler_file.write(scaler_response.content)
        temp_scaler_file_path = temp_scaler_file.name  # Store the path

    # Load the scaler
    scaler = joblib.load(temp_scaler_file_path)

    # Clean up the temporary scaler file
    os.remove(temp_scaler_file_path)

    return model, scaler

def predict_with_model(model, scaler, df, window_size=60):
    """
    Make predictions using a trained model.
    
    Parameters:
        model: The trained Keras model.
        scaler: The scaler used for preprocessing.
        df (DataFrame): The data to make predictions on.
        window_size (int): The size of the window to use for predictions.
    
    Returns:
        predictions (ndarray): The predicted values.
    """
    # Ensure the data is scaled
    scaled_data = scaler.transform(df.drop(columns=['date']))
    
    # Prepare the input data
    X = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
    
    X = np.array(X)
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions

def save_model(model, scaler, model_name, scaler_name, model_type="generic"):
    """
    Save a trained model and its corresponding scaler to disk.
    
    Parameters:
        model: The trained Keras model.
        scaler: The scaler used for preprocessing.
        model_name (str): The name to save the model file as.
        scaler_name (str): The name to save the scaler file as.
        model_type (str): Either "generic" for the generic model or "specific" for ticker-specific models.
    """
    if model_type == "generic":
        model_dir = "AITraining/models/lstm_generic_models"
    else:
        model_dir = "AITraining/models/lstm_multifeature_models"

    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, model_name)
    scaler_path = os.path.join(model_dir, scaler_name)
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model and scaler saved to {model_dir} as {model_name} and {scaler_name}.")

def load_saved_model_and_scaler(ticker, model_type="specific"):
    """
    Load a saved model and its corresponding scaler from disk.
    
    Parameters:
        ticker (str): The stock ticker.
        model_type (str): Either "generic" for the generic model or "specific" for ticker-specific models.
    
    Returns:
        model, scaler: The loaded Keras model and the scaler used during training.
    """
    if model_type == "generic":
        model_name = "generic_lstm_multifeature.h5"
        scaler_name = "generic_scaler_multifeature.pkl"
        model_dir = "AITraining/models/lstm_generic_models"
    else:
        model_name = f"{ticker}_lstm_multifeature.h5"
        scaler_name = f"{ticker}_scaler_multifeature.pkl"
        model_dir = "AITraining/models/lstm_multifeature_models"
    
    model_path = os.path.join(model_dir, model_name)
    scaler_path = os.path.join(model_dir, scaler_name)
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        raise FileNotFoundError(f"Model or scaler not found for {ticker if ticker else 'generic'} in {model_dir}")
