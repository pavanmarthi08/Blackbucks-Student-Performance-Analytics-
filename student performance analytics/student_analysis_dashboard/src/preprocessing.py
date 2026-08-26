import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor(numeric_features, categorical_features):
    """
    Creates a scikit-learn ColumnTransformer for preprocessing.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def prepare_data(df, target_col='Exam_Score', drop_cols=['Student_ID']):
    """
    Prepares data by separating features and target, and identifying column types.
    """
    # Drop identifiers or irrelevant columns
    df_clean = df.drop(columns=drop_cols, errors='ignore')
    
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return X, y, numeric_features, categorical_features

def split_and_preprocess(df, target_col='Exam_Score', test_size=0.2, random_state=42):
    """
    Splits data and returns un-preprocessed DataFrames along with the fitted preprocessor.
    This is useful for pipelines that bundle the preprocessor with the model.
    """
    X, y, num_cols, cat_cols = prepare_data(df, target_col)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    preprocessor = get_preprocessor(num_cols, cat_cols)
    
    return X_train, X_test, y_train, y_test, preprocessor
