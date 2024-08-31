# src/utils/data_loader.py

import pandas as pd

def load_data(outlet_info_path, training_data_path, test_data_path):
    outlet_info = pd.read_csv(outlet_info_path)
    training_data = pd.read_csv(training_data_path)
    test_data = pd.read_csv(test_data_path)
    return outlet_info, training_data, test_data