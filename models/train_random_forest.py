# src/models/train_random_forest.py
import sys
import os

# Get the project root directory (two levels up from the current directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the project root directory to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib


def train_random_forest(X_train, y_train):
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_split, y_train_split)
    
    # Validate the model
    val_predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_predictions)
    print(f'MAE on validation set: {mae}')
    
    # Save the model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/random_forest_model.pkl')
    print("Random Forest Model training complete. Model saved.")
    
    return model
