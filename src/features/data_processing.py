# src/features/data_processing.py

import pandas as pd
import numpy as np

def process_data_sources(outlet_info, training_data, test_data):
    # Merging outlet information with training data
    training_data = training_data.merge(outlet_info, on='store', how='left')
    test_data = test_data.merge(outlet_info, on='store', how='left')

    return training_data, test_data

def generate_primary_keys(data):
    # Generate primary keys by combining store, department, and date
    data['primary_key'] = data['store'] + '|' + data['item_dept'] + '|' + data['date_id'].astype(str)
    return data

def create_target_variable(data):
    # Target variable is the item_qty for forecasting
    data['target'] = data['item_qty']
    return data
