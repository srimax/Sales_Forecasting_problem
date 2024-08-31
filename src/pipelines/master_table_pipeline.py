# src/pipelines/master_table_pipeline.py
import sys
import os

# Get the project root directory (two levels up from the current directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the project root directory to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from src.features.sales_features import add_sales_features
from src.features.item_features import add_item_features
from src.features.time_features import add_time_features
from src.features.outlet_features import add_outlet_features

def create_master_table(processed_training_data_path, processed_test_data_path):
    # Load processed data
    training_data = pd.read_csv(processed_training_data_path)
    test_data = pd.read_csv(processed_test_data_path)

    # Debugging: Check the initial columns of the loaded data
    print("Initial training data columns:", training_data.columns.tolist())
    print("Initial test data columns:", test_data.columns.tolist())

    # Create the 'target' column first, before any additional processing
    if 'item_qty' in training_data.columns:
        training_data['target'] = training_data['item_qty']
    else:
        raise KeyError("The column 'item_qty' does not exist in the training data.")
    
    # For the test data, even if the 'item_qty' column is missing, we should create the 'target' column
    if 'item_qty' in test_data.columns:
        test_data['target'] = test_data['item_qty']
    else:
        test_data['target'] = None  # Placeholder, as test data might not have this column

    # Debugging: Verify that the 'target' column was added
    print("Training data columns after adding target:", training_data.columns.tolist())
    print("Test data columns after adding target:", test_data.columns.tolist())

    # Now apply additional feature engineering steps
    training_data = add_sales_features(training_data)
    training_data = add_item_features(training_data)
    training_data = add_time_features(training_data)
    training_data = add_outlet_features(training_data)

    test_data = add_sales_features(test_data)
    test_data = add_item_features(test_data)
    test_data = add_time_features(test_data)
    test_data = add_outlet_features(test_data)

    # Ensure the data directory exists
    output_dir = os.path.join(os.path.dirname(processed_training_data_path), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Save the master table
    training_data.to_csv(os.path.join(output_dir, 'master_training_data.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'master_test_data.csv'), index=False)

    print("Master table created and saved.")

    return training_data, test_data
