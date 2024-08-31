# src/models/train_prophet_model.py
import sys
import os

# Get the project root directory (two levels up from the current directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the project root directory to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)

from prophet import Prophet
import pandas as pd
import os
import joblib

def prepare_prophet_data(training_data):
    # Prophet requires data in a specific format with columns 'ds' and 'y'
    df = training_data[['date_id', 'target']].copy()
    df.rename(columns={'date_id': 'ds', 'target': 'y'}, inplace=True)
    return df

def train_prophet_model(training_data):
    # Prepare data for Prophet
    prophet_data = prepare_prophet_data(training_data)
    
    # Initialize and train Prophet model
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_data)
    
    # Save the model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/prophet_model.pkl')
    print("Prophet Model training complete. Model saved.")
    
    return model
