# src/models/tune_random_forest.py
import sys
import os

# Get the project root directory (two levels up from the current directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the project root directory to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib


def tune_random_forest(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    
    # Define hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Save the best model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_model, '../models/random_forest_best_model.pkl')
    print(f"Best Model Parameters: {grid_search.best_params_}")
    print("Best Random Forest Model training complete. Model saved.")
    
    return best_model
