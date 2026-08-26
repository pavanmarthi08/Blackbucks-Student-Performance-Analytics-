import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import load_or_generate_data
from src.preprocessing import split_and_preprocess

def evaluate_model(name, model, X_test, y_test):
    """Evaluates a model and returns a dictionary of metrics."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    
    print(f"--- {name} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}\n")
    
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

def train_and_evaluate(data_path="data/student_performance.csv", model_dir="models"):
    """
    Trains multiple models, evaluates them, and saves the best one (Random Forest).
    """
    print("Loading data...")
    df = load_or_generate_data(data_path)
    
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df)
    
    # Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    pipelines = {}
    
    print("Training models...\n")
    for name, model in models.items():
        # Create a pipeline that bundles preprocessing and the model
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
        
        clf.fit(X_train, y_train)
        metrics = evaluate_model(name, clf, X_test, y_test)
        results.append(metrics)
        pipelines[name] = clf
        
    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    print("Model Comparison:")
    print(results_df.to_string(index=False))
    
    # We choose Random Forest as the main model as per requirements
    final_model = pipelines['Random Forest']
    
    # Save the model with compression
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'student_performance_model.pkl')
    joblib.dump(final_model, model_path, compress=3)
    print(f"\nFinal Random Forest model saved to {model_path}")
    
    # Extract Feature Importances
    # Get feature names after preprocessing
    rf = final_model.named_steps['model']
    prep = final_model.named_steps['preprocessor']
    
    cat_encoder = prep.named_transformers_['cat'].named_steps['onehot']
    cat_features_in = prep.transformers_[1][2]
    num_features_in = prep.transformers_[0][2]
    
    # categorical feature names
    cat_features_out = cat_encoder.get_feature_names_out(cat_features_in)
    
    all_features = list(num_features_in) + list(cat_features_out)
    importances = rf.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10).to_string(index=False))
    
    return final_model, importance_df

if __name__ == "__main__":
    # Adjust paths if running directly from src folder
    data_path = "../data/student_performance.csv"
    model_dir = "../models"
    train_and_evaluate(data_path, model_dir)
