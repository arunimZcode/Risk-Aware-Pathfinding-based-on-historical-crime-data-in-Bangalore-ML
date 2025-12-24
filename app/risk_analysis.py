import torch
import torch.nn as nn
import joblib
import numpy as np
import os

# ---------------------------------------------------------
# Neural Network Architecture
# Matches the specific layer naming in your latest .pth file
# ---------------------------------------------------------
class CrimeRiskNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=1):
        super(CrimeRiskNet, self).__init__()
        # Matches Sequential structure: net.0 (Linear), net.2 (Linear), net.4 (Linear)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),   # net.0
            nn.ReLU(),                            # net.1
            nn.Linear(hidden_size, 32),           # net.2
            nn.ReLU(),                            # net.3
            nn.Linear(32, output_size),           # net.4
            nn.Sigmoid()                          # Output mapping to 0-1 range
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# Global Loaders & Path Resolution
# ---------------------------------------------------------
def find_file(filename):
    """Checks current directory and /data/ folder for the required file."""
    paths = [filename, os.path.join('data', filename), os.path.join('..', 'data', filename)]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

MODEL_PATH = find_file('crime_risk_model.pth')
# location_kmeans.pkl is no longer strictly needed if features are raw lat/lon
KMEANS_PATH = find_file('location_kmeans.pkl') 
RISK_MULTIPLIER = 1000
DAY_MAP = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

# Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
risk_model = None

try:
    # Initialize and Load PyTorch Model with 4 inputs
    risk_model = CrimeRiskNet(input_size=4).to(device)
    
    if MODEL_PATH:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        risk_model.load_state_dict(state_dict)
        risk_model.eval() 
        print(f"Successfully loaded PyTorch model from {MODEL_PATH} on {device}")
    else:
        print(f"Warning: crime_risk_model.pth not found. System will use default risk values.")

except Exception as e:
    print(f"Error initializing models: {e}")

def get_risk_penalty(lat: float, lon: float, hour: int, day_of_week: str) -> float:
    """
    Predicts a risk score (0.0 to 1.0) using 4 input features:
    [lat, lon, hour, day_index]
    """
    if risk_model is None:
        return 0.1 # Minimal baseline safety fallback

    try:
        # 1. Map Day to Index
        day_idx = DAY_MAP.get(day_of_week.capitalize(), 0)
        
        # 2. Construct Feature Vector [lat, lon, hour, day_idx]
        features = [float(lat), float(lon), float(hour), float(day_idx)]
        input_tensor = torch.FloatTensor(features).to(device).unsqueeze(0)

        # 3. Model Inference
        with torch.no_grad():
            prediction = risk_model(input_tensor)
            risk_score = prediction.item()
        
        return float(np.clip(risk_score, 0.0, 1.0))
        
    except Exception as e:
        return 0.05 # Fallback on error