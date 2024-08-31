import os
import sys

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the project root directory to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)

from src.pipelines.model_pipeline import main as run_model_pipeline
from src.pipelines.master_table_pipeline import create_master_table
from src.pipelines.evaluation import evaluate_model

def main():
    # Step 1: Create the Master Table
    processed_training_data_path = os.path.join(project_root, 'data', 'processed_training_data.csv')
    processed_test_data_path = os.path.join(project_root, 'data', 'processed_test_data.csv')
    create_master_table(processed_training_data_path, processed_test_data_path)
    
    # Step 2: Train the Model and Generate Predictions
    run_model_pipeline()

    # Step 3: Evaluate the Model
    evaluate_model()

if __name__ == "__main__":
    main()
