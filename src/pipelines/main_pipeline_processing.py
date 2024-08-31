# src/pipelines/main_pipeline.py

import sys
import os

# Get the project root directory (two levels up from the current directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the project root directory to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.pipelines.master_table_pipeline import create_master_table
from src.pipelines.model_pipeline import train_model
from src.pipelines.evaluation import evaluate_model

def main():
    # Construct absolute paths to the data files
    processed_training_data_path = os.path.join(project_root, 'data/processed_training_data.csv')
    processed_test_data_path = os.path.join(project_root, 'data/processed_test_data.csv')
    
    # Step 1: Create the master table
    training_data, test_data = create_master_table(processed_training_data_path, processed_test_data_path)
    
    # Step 2: Train the model
    model = train_model(os.path.join(project_root, 'data/master_training_data.csv'))
    
    # Step 3: Evaluate the model
    evaluate_model(os.path.join(project_root, 'data/master_test_data.csv'), os.path.join(project_root, 'models/model.pkl'))

if __name__ == "__main__":
    main()
