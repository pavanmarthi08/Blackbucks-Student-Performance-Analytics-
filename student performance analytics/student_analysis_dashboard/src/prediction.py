import joblib
import pandas as pd
import os

def load_model(model_path="models/student_performance_model.pkl"):
    """Loads the trained model pipeline."""
    candidates = [
        model_path,
        os.path.join(os.path.dirname(__file__), "..", "..", model_path),
        os.path.join(os.path.dirname(__file__), "..", model_path),
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "student_performance_model.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "models", "student_performance_model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"Model not found at {model_path}. Please run training first.")

def predict_performance(model, input_data):
    """
    Predicts student performance based on input dictionary.
    Returns a tuple: (predicted_score, risk_category, recommendations)
    """
    df = pd.DataFrame([input_data])
    score = model.predict(df)[0]
    
    # Categorize Risk
    if score < 60:
        risk = "High Risk"
        recommendations = [
            "Immediate academic intervention recommended.",
            "Schedule a meeting with a tutor or academic advisor.",
            "Review study habits and try to increase study hours.",
            "Focus on improving daily attendance."
        ]
    elif score < 75:
        risk = "Medium Risk"
        recommendations = [
            "Monitor progress carefully.",
            "Encourage more consistent study habits.",
            "Check for specific subjects causing difficulty."
        ]
    else:
        risk = "Low Risk"
        recommendations = [
            "Student is performing well.",
            "Maintain current study habits and attendance.",
            "Consider advanced or extracurricular challenges to maintain motivation."
        ]
        
    return round(score, 2), risk, recommendations

def extract_feature_importance(model):
    """Extracts feature importances from the loaded random forest pipeline."""
    try:
        rf = model.named_steps['model']
        prep = model.named_steps['preprocessor']
        
        cat_encoder = prep.named_transformers_['cat'].named_steps['onehot']
        cat_features_in = prep.transformers_[1][2]
        num_features_in = prep.transformers_[0][2]
        
        cat_features_out = cat_encoder.get_feature_names_out(cat_features_in)
        all_features = list(num_features_in) + list(cat_features_out)
        
        importances = rf.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        return importance_df
    except Exception as e:
        print(f"Could not extract feature importances: {e}")
        return pd.DataFrame()
