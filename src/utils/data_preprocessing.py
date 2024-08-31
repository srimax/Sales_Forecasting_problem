# src/utils/data_preprocessing.py

import pandas as pd

def clean_data(training_data):
    # Drop duplicates if any
    training_data.drop_duplicates(inplace=True)
    
    # Handle missing values - for this example, we'll fill missing item_qty with 0
    training_data['item_qty'].fillna(0, inplace=True)
    
    # Convert date_id to datetime format
    training_data['date_id'] = pd.to_datetime(training_data['date_id'])
    
    return training_data
