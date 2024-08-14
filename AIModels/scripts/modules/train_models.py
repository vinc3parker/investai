import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# LSTM Model class
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out
    
# Training Function
def train_lstm_model(X_train, y_train, X_val, y_val, ticker):
    """
    Train an LSTM model for a given stock ticker.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - ticker: The stock ticker symbol.

    Returns:
    - model: The trained LSTM model.
    """
    # Ensure that the inputs are PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    print(f"Shape of X_train: {X_train.shape}")
    model = StockLSTM(input_size=X_train.shape[2], hidden_size=50, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        val_output = model(X_val)
        val_loss = criterion(val_output.squeeze(), y_val).item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Validation Loss: {val_loss}")

    return model

# Saving Function
def save_model_and_scaler(model, scaler, ticker):
    """
    Save the trained model and scaler.

    Parameters:
    - model: The trained LSTM model.
    - scaler: The MinMaxScaler used for feature scaling.
    - ticker: The stock ticker symbol.
    """
    model_dir = "models/lstm_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{ticker}_lstm_model.pt")
    scaler_path = os.path.join(model_dir, f"{ticker}_scaler.pkl")
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model and scaler saved for {ticker} at {model_dir}")

# Loading Function
def load_model_and_scaler(ticker):
    """
    Load a saved model and scaler for a given ticker.

    Parameters:
    - ticker: The stock ticker symbol.

    Returns:
    - model: The loaded LSTM model.
    - scaler: The loaded MinMaxScaler.
    """
    model_dir = "models/lstm_models"
    model_path = os.path.join(model_dir, f"{ticker}_lstm_model.pt")
    scaler_path = os.path.join(model_dir, f"{ticker}_scaler.pkl")

    model = StockLSTM(input_size=8, hidden_size=50, output_size=1)  # Adjust input_size if needed
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaler = joblib.load(scaler_path)
    print(f"Model and scaler loaded for {ticker} from {model_dir}")

    return model, scaler