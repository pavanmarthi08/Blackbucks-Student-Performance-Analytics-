import pandas as pd
import numpy as np
import os

def generate_realistic_student_data(num_students=5000, random_state=42):
    """
    Generates a realistic student performance dataset with logical correlations
    between features and the target variable (Exam Score).
    """
    np.random.seed(random_state)
    
    # 1. Base Demographics
    student_ids = [f"STU{str(i).zfill(5)}" for i in range(1, num_students + 1)]
    age = np.random.randint(15, 22, size=num_students)
    gender = np.random.choice(['Male', 'Female', 'Other'], size=num_students, p=[0.48, 0.48, 0.04])
    family_income = np.random.choice(['Low', 'Medium', 'High'], size=num_students, p=[0.3, 0.5, 0.2])
    distance_from_school = np.random.choice(['Near', 'Moderate', 'Far'], size=num_students, p=[0.4, 0.4, 0.2])
    
    # 2. Behavioral & Academic Features
    # Base score determined by some intrinsic factors and noise
    base_score = np.random.normal(60, 10, size=num_students)
    
    # Attendance: 50% to 100%
    attendance = np.clip(np.random.normal(85, 10, size=num_students), 50, 100)
    
    # Study Hours per week: 0 to 40
    study_hours = np.clip(np.random.normal(15, 8, size=num_students), 0, 40)
    
    # Previous Scores: 0 to 100, correlated with base
    previous_scores = np.clip(base_score + np.random.normal(0, 5, size=num_students), 0, 100)
    
    sleep_hours = np.clip(np.random.normal(7, 1.5, size=num_students), 4, 10)
    motivation = np.random.choice(['Low', 'Medium', 'High'], size=num_students, p=[0.2, 0.5, 0.3])
    parental_involvement = np.random.choice(['Low', 'Medium', 'High'], size=num_students, p=[0.25, 0.5, 0.25])
    internet_access = np.random.choice(['Yes', 'No'], size=num_students, p=[0.85, 0.15])
    tutoring = np.random.choice(['Yes', 'No'], size=num_students, p=[0.3, 0.7])
    extracurricular = np.random.choice(['Yes', 'No'], size=num_students, p=[0.4, 0.6])
    learning_difficulties = np.random.choice(['Yes', 'No'], size=num_students, p=[0.1, 0.9])
    academic_support = np.random.choice(['Yes', 'No'], size=num_students, p=[0.2, 0.8])

    # Assignment Performance: Correlated with study hours
    assignment_performance = np.clip(50 + (study_hours * 1.2) + np.random.normal(0, 10, size=num_students), 0, 100)

    # 3. Calculate Final Exam Score based on correlations
    exam_score = previous_scores * 0.4 + \
                 (attendance / 100) * 20 + \
                 (study_hours * 0.5) + \
                 (assignment_performance * 0.2)
                 
    # Modifiers
    modifier = np.zeros(num_students)
    
    # Motivation effect
    modifier[motivation == 'High'] += 5
    modifier[motivation == 'Low'] -= 5
    
    # Sleep effect (optimal around 7-8 hours)
    sleep_penalty = np.abs(sleep_hours - 7.5) * -1.5
    modifier += sleep_penalty
    
    # Other categorical effects
    modifier[tutoring == 'Yes'] += 4
    modifier[family_income == 'High'] += 3
    modifier[family_income == 'Low'] -= 2
    modifier[internet_access == 'Yes'] += 3
    modifier[learning_difficulties == 'Yes'] -= 6
    modifier[(learning_difficulties == 'Yes') & (academic_support == 'Yes')] += 4 # Support helps mitigate
    
    exam_score += modifier
    
    # Add random noise
    exam_score += np.random.normal(0, 4, size=num_students)
    
    # Clip to 0-100
    exam_score = np.clip(exam_score, 0, 100)

    # Create DataFrame
    df = pd.DataFrame({
        'Student_ID': student_ids,
        'Age': age,
        'Gender': gender,
        'Study_Hours_per_Week': np.round(study_hours, 1),
        'Attendance_Rate': np.round(attendance, 1),
        'Previous_Score': np.round(previous_scores, 1),
        'Assignment_Performance': np.round(assignment_performance, 1),
        'Sleep_Hours_per_Night': np.round(sleep_hours, 1),
        'Motivation_Level': motivation,
        'Parental_Involvement': parental_involvement,
        'Internet_Access': internet_access,
        'Tutoring': tutoring,
        'Extracurricular_Activities': extracurricular,
        'Family_Income': family_income,
        'Distance_from_School': distance_from_school,
        'Learning_Difficulties': learning_difficulties,
        'Academic_Support': academic_support,
        'Exam_Score': np.round(exam_score, 1)
    })
    
    # Introduce some realistic missing values (MCAR)
    # E.g., ~2% missing data in certain columns
    cols_with_missing = ['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Parental_Involvement']
    for col in cols_with_missing:
        mask = np.random.rand(num_students) < 0.02
        df.loc[mask, col] = np.nan
        
    return df

def load_or_generate_data(data_path="data/student_performance.csv"):
    """
    Loads data if it exists, otherwise generates it, saves it, and returns it.
    """
    candidates = [
        data_path,
        os.path.join(os.path.dirname(__file__), "..", "..", data_path),
        os.path.join(os.path.dirname(__file__), "..", data_path),
        os.path.join(os.path.dirname(__file__), "..", "data", "student_performance.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading existing dataset from {path}")
            return pd.read_csv(path)
    
    print(f"Dataset not found at {data_path}. Generating synthetic data...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df = generate_realistic_student_data()
    df.to_csv(data_path, index=False)
    print("Data generated and saved.")
    return df
    print("\nSample Data:")
    print(df.head())
