#!/bin/bash

Safe Route Planner - Startup Script

echo "------------------------------------------------"
echo "Initializing Safe Route Planner Environment..."
echo "------------------------------------------------"

1. Create necessary directories if they don't exist

mkdir -p data
mkdir -p app

2. Check for required model files in the data directory

if [ ! -f "data/crime_risk_model.pth" ]; then
echo "Warning: data/crime_risk_model.pth not found!"
echo "Please ensure your PyTorch model is placed in the data/ folder."
fi

if [ ! -f "data/location_kmeans.pkl" ]; then
echo "Warning: data/location_kmeans.pkl not found!"
echo "Please ensure your K-Means clusterer is placed in the data/ folder."
fi

3. Install dependencies

It is recommended to use a virtual environment

echo "Installing/Updating dependencies..."
pip install --upgrade pip
pip install fastapi uvicorn torch pandas numpy osmnx networkx joblib geopy

4. Start the FastAPI server using Uvicorn

echo "------------------------------------------------"
echo "Starting FastAPI server on https://www.google.com/search?q=http://0.0.0.0:8000"
echo "------------------------------------------------"

We run main.app where 'main' is the filename and 'app' is the FastAPI instance

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


to run:
python -m uvicorn app.main:app --reload
then click on the index.html on the folder to open in browser.