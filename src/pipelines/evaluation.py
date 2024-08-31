import pandas as pd
import os

# Define the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    return (abs((y_true - y_pred) / y_true)).mean() * 100

def evaluate_predictions(predictions, granularity):
    """
    Evaluate the predictions by the specified granularity.
    Possible granularities: ['store', 'date_id', 'item_dept']
    """
    grouped_data = predictions.groupby(granularity).agg({
        'item_qty': 'sum',
        'predicted_item_qty': 'sum'
    }).reset_index()

    mape = calculate_mape(grouped_data['item_qty'], grouped_data['predicted_item_qty'])
    return mape

def evaluate_model():
    # Load the predictions
    predictions_path = os.path.join(project_root, 'data', 'february_predictions.csv')
    predictions = pd.read_csv(predictions_path)

    # MAPE for Store | Department | Date
    mape_store_dept_date = evaluate_predictions(predictions, ['store', 'item_dept', 'date_id'])
    print(f"MAPE (Store | Department | Date): {mape_store_dept_date:.2f}%")
    
    # MAPE for Store | Date
    mape_store_date = evaluate_predictions(predictions, ['store', 'date_id'])
    print(f"MAPE (Store | Date): {mape_store_date:.2f}%")

    return mape_store_dept_date, mape_store_date

if __name__ == "__main__":
    evaluate_model()
