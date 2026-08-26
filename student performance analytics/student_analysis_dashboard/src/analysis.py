import pandas as pd
import numpy as np

def get_kpis(df):
    """Calculates key performance indicators for the overview page."""
    kpis = {
        "Total Students": len(df),
        "Average Score": round(df["Exam_Score"].mean(), 1),
        "Highest Score": round(df["Exam_Score"].max(), 1),
        "Lowest Score": round(df["Exam_Score"].min(), 1),
        "Average Attendance": round(df["Attendance_Rate"].mean(), 1),
        "At-Risk Students": len(df[df["Exam_Score"] < 60])
    }
    return kpis

def get_correlation_matrix(df):
    """Returns the correlation matrix for numerical features."""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    return numeric_df.corr()

def get_risk_distribution(df):
    """Categorizes students into risk levels and returns counts."""
    conditions = [
        (df['Exam_Score'] < 60),
        (df['Exam_Score'] >= 60) & (df['Exam_Score'] < 75),
        (df['Exam_Score'] >= 75)
    ]
    choices = ['High Risk', 'Medium Risk', 'Low Risk']
    df['Risk_Level'] = np.select(conditions, choices, default='Unknown')
    return df['Risk_Level'].value_counts()
