# src/features/time_features.py
import pandas as pd

def add_time_features(data):
    # Ensure that the 'date_id' column is in datetime format
    data['date_id'] = pd.to_datetime(data['date_id'], errors='coerce')

    # Example: Extract time-based features
    data['day_of_week'] = data['date_id'].dt.dayofweek
    data['month'] = data['date_id'].dt.month
    data['quarter'] = data['date_id'].dt.quarter
    
    return data
