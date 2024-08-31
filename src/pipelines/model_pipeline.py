import sys
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder

# Get the project root directory (two levels up from the current directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the project root directory to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)

from models.train_random_forest import train_random_forest
from models.train_prophet_model import train_prophet_model
from models.tune_random_forest import tune_random_forest


def encode_categorical_features(X_train, X_test, categorical_columns):
    """Encode categorical features ensuring consistent encoding between training and test data."""
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Fit the encoder on the training data and transform both train and test data
    X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
    X_test_encoded = encoder.transform(X_test[categorical_columns])
    
    # Convert the encoded arrays back to DataFrames
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Reset index to ensure the original DataFrame's indices match
    X_train_encoded_df.index = X_train.index
    X_test_encoded_df.index = X_test.index
    
    # Drop the original categorical columns and concatenate the encoded columns
    X_train = X_train.drop(columns=categorical_columns).join(X_train_encoded_df)
    X_test = X_test.drop(columns=categorical_columns).join(X_test_encoded_df)
    
    return X_train, X_test

def load_and_prepare_data():
    data_dir = os.path.join(project_root, 'data')
    master_training_data_path = os.path.join(data_dir, 'master_training_data.csv')
    master_test_data_path = os.path.join(data_dir, 'master_test_data.csv')

    training_data = pd.read_csv(master_training_data_path)
    test_data = pd.read_csv(master_test_data_path)

    print("Columns in training data:", training_data.columns.tolist())
    print("Columns in test data:", test_data.columns.tolist())

    if 'target' not in test_data.columns:
        print("The 'target' column is missing in test data. Adding an empty 'target' column.")
        test_data['target'] = None

    training_data['target'].fillna(0, inplace=True)
    test_data['target'].fillna(0, inplace=True)

    relevant_departments = ['Beverages', 'Grocery', 'Household']
    relevant_outlets = ['XYZ', 'ABC']

    filtered_training_data = training_data[
        (training_data['item_dept'].isin(relevant_departments)) & 
        (training_data['store'].isin(relevant_outlets))
    ]

    filtered_test_data = test_data[
        (test_data['item_dept'].isin(relevant_departments)) & 
        (test_data['store'].isin(relevant_outlets))
    ]

    columns_to_drop = ['primary_key', 'date_id', 'store', 'item_dept']
    target_column = 'target'

    print("Filtered test data columns:", filtered_test_data.columns.tolist())

    if target_column not in filtered_test_data.columns:
        raise KeyError(f"The target column '{target_column}' does not exist in the test data.")

    X_train = filtered_training_data.drop(columns=columns_to_drop + [target_column], axis=1)
    y_train = filtered_training_data[target_column]
    
    X_test = filtered_test_data.drop(columns=columns_to_drop + [target_column], axis=1)
    y_test = filtered_test_data[target_column]

    categorical_columns = ['profile', 'size', 'is_large', 'is_high_profile']
    
    X_train, X_test = encode_categorical_features(X_train, X_test, categorical_columns)

    return X_train, y_train, X_test, y_test, filtered_test_data

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'model.pkl'))
    print(f"Model training complete. Model saved to {os.path.join(models_dir, 'model.pkl')}")
    
    return model

def validate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    y_test = pd.Series(y_test)

    valid_idx = y_test.notna() & pd.Series(predictions).notna()
    y_test = y_test[valid_idx]
    predictions = predictions[valid_idx]

    if y_test.empty:
        print("Warning: No valid data points found for validation after filtering out NaNs.")
        return

    mape = mean_absolute_percentage_error(y_test, predictions)
    print(f'MAPE on test set: {mape}')

def predict_sales(model, february_test_data, categorical_columns):
    # Drop unnecessary columns except 'item_qty' and others required for grouping
    features = february_test_data.drop(columns=['date_id', 'store', 'item_dept', 'target', 'primary_key'])

    # Encode categorical features in the test data
    features, _ = encode_categorical_features(features, features, categorical_columns)

    # Generate predictions
    predictions = model.predict(features)
    
    # Add predictions to the test data
    february_test_data['predicted_item_qty'] = predictions
    
    return february_test_data

def aggregate_predictions(predictions):
    aggregated_predictions = predictions.groupby(['date_id', 'store', 'item_dept']).agg({
        'item_qty': 'sum',  # Add item_qty here
        'predicted_item_qty': 'sum'
    }).reset_index()
    
    return aggregated_predictions

def main():
    X_train, y_train, X_test, y_test, filtered_test_data = load_and_prepare_data()

    categorical_columns = ['profile', 'size', 'is_large', 'is_high_profile']

    # Train the model
    #model = train_model(X_train, y_train)
    model = train_random_forest(X_train, y_train)
    #model = train_prophet_model(X_train)
    #model = tune_random_forest(X_train, y_train)
    validate_model(model, X_test, y_test)
    february_predictions = predict_sales(model, filtered_test_data, categorical_columns)
    final_predictions = aggregate_predictions(february_predictions)
    
    predictions_path = os.path.join(project_root, 'data', 'february_predictions.csv')
    final_predictions.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    main()
