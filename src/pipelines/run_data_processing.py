# run_data_processing.py
import sys
import os

# Get the project root directory (two levels up from the current directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the project root directory to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd

from src.utils.data_loader import load_data
from src.features.data_processing import process_data_sources, generate_primary_keys, create_target_variable

# Paths to the data files
outlet_info_path = os.path.join(project_root, 'data/outlet_info.csv')
cleaned_training_data_path = os.path.join(project_root, 'data/cleaned_training_data.csv')
cleaned_test_data_path = os.path.join(project_root, 'data/cleaned_test_data.csv')

def main():
    # Load the cleaned data
    outlet_info = pd.read_csv(outlet_info_path)
    training_data = pd.read_csv(cleaned_training_data_path)
    test_data = pd.read_csv(cleaned_test_data_path)

    # Process the data
    training_data, test_data = process_data_sources(outlet_info, training_data, test_data)

    # Generate primary keys
    training_data = generate_primary_keys(training_data)
    test_data = generate_primary_keys(test_data)

    # Create target variable
    training_data = create_target_variable(training_data)

    # Save the processed data for future use
    training_data.to_csv(os.path.join(project_root, 'data/processed_training_data.csv'), index=False)
    test_data.to_csv(os.path.join(project_root, 'data/processed_test_data.csv'), index=False)
    print("Data processing complete. Processed files saved.")

if __name__ == "__main__":
    main()
