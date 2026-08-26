# Algorithm: Student Performance Prediction System

## Overview
This system predicts a student's final exam score from academic, behavioral, and
demographic features, classifies the student into a risk category, and generates
supporting analytics. It consists of five stages: **Data Generation/Loading**,
**Preprocessing**, **Model Training**, **Prediction**, and **Analysis**.

---

## 1. Data Acquisition Algorithm
**Purpose:** Obtain a student performance dataset, generating synthetic data if none exists.

```
ALGORITHM LoadOrGenerateData(data_path)
    FOR each candidate path in [data_path, and 3 relative fallback paths]:
        IF file exists at candidate:
            RETURN data loaded from CSV
    // No existing dataset found
    df ← GenerateSyntheticData(num_students = 5000, seed = 42)
    Save df to data_path as CSV
    RETURN df
END
```

**Synthetic Data Generation sub-algorithm:**
```
ALGORITHM GenerateSyntheticData(n, seed)
    Set random seed

    1. Sample demographic features:
         Age ~ Uniform[15,22)
         Gender, Family_Income, Distance_from_School ~ weighted categorical draws

    2. Sample behavioral/academic base features:
         base_score        ~ Normal(60, 10)
         Attendance_Rate   ~ clip(Normal(85, 10), 50, 100)
         Study_Hours       ~ clip(Normal(15, 8), 0, 40)
         Previous_Score    ~ clip(base_score + Normal(0,5), 0, 100)
         Sleep_Hours       ~ clip(Normal(7, 1.5), 4, 10)
         Motivation, Parental_Involvement, Internet_Access, Tutoring,
         Extracurricular, Learning_Difficulties, Academic_Support
                            ~ weighted categorical draws
         Assignment_Performance ~ clip(50 + 1.2*Study_Hours + Normal(0,10), 0, 100)

    3. Compute base Exam_Score as a weighted linear combination:
         Exam_Score = 0.4*Previous_Score
                     + 0.20*Attendance_Rate
                     + 0.5*Study_Hours
                     + 0.2*Assignment_Performance

    4. Apply categorical modifiers (additive bonuses/penalties):
         +5 if Motivation = High,  -5 if Motivation = Low
         Sleep penalty = -1.5 * |Sleep_Hours - 7.5|   (peak at 7.5h)
         +4 if Tutoring = Yes
         +3 if Family_Income = High, -2 if Low
         +3 if Internet_Access = Yes
         -6 if Learning_Difficulties = Yes
         +4 if (Learning_Difficulties = Yes AND Academic_Support = Yes)

    5. Add random Gaussian noise ~ Normal(0, 4) to Exam_Score
    6. Clip Exam_Score to [0, 100]
    7. Randomly null out ~2% of values in 3 columns (simulate missing data)
    8. RETURN assembled DataFrame
END
```

---

## 2. Preprocessing Algorithm
**Purpose:** Prepare raw features for model consumption — imputing missing
values, scaling numeric features, and encoding categorical features.

```
ALGORITHM PrepareData(df, target_col, drop_cols)
    df_clean ← df with drop_cols (e.g., Student_ID) removed
    X ← df_clean without target_col
    y ← df_clean[target_col]
    numeric_features   ← columns of X with dtype int/float
    categorical_features ← columns of X with dtype object/category
    RETURN X, y, numeric_features, categorical_features
END

ALGORITHM BuildPreprocessor(numeric_features, categorical_features)
    numeric_pipeline    ← Impute(median) → StandardScale
    categorical_pipeline ← Impute(most_frequent) → OneHotEncode(ignore unknown)
    RETURN ColumnTransformer combining both pipelines by column group
END

ALGORITHM SplitAndPreprocess(df, target_col, test_size=0.2, seed=42)
    X, y, num_cols, cat_cols ← PrepareData(df, target_col)
    X_train, X_test, y_train, y_test ← train_test_split(X, y, test_size, seed)
    preprocessor ← BuildPreprocessor(num_cols, cat_cols)
    RETURN X_train, X_test, y_train, y_test, preprocessor
END
```

---

## 3. Model Training Algorithm
**Purpose:** Train multiple regression models, compare them, and persist the best one.

```
ALGORITHM TrainAndEvaluate(data_path, model_dir)
    df ← LoadOrGenerateData(data_path)
    X_train, X_test, y_train, y_test, preprocessor ← SplitAndPreprocess(df)

    candidate_models ← {
        "Linear Regression"  : LinearRegression(),
        "Random Forest"      : RandomForestRegressor(100 trees, seed=42),
        "Gradient Boosting"  : GradientBoostingRegressor(100 estimators, seed=42)
    }

    FOR each (name, model) in candidate_models:
        pipeline ← Pipeline(preprocessor → model)
        pipeline.fit(X_train, y_train)
        preds ← pipeline.predict(X_test)
        metrics[name] ← { MAE, RMSE, R2 } computed from (y_test, preds)
        store pipeline[name]

    Rank models by R2 (descending)
    final_model ← pipeline["Random Forest"]     // selected as production model
    Save final_model to model_dir/student_performance_model.pkl (compressed)

    Extract feature importances from final_model's Random Forest step
    RETURN final_model, feature_importance_table
END
```

---

## 4. Prediction Algorithm
**Purpose:** Score a new student record and translate it into an actionable risk category.

```
ALGORITHM PredictPerformance(model, input_data)
    df ← single-row DataFrame from input_data
    score ← model.predict(df)[0]

    IF score < 60:
        risk ← "High Risk"
        recommendations ← [intervention, tutor meeting, study hours, attendance]
    ELSE IF score < 75:
        risk ← "Medium Risk"
        recommendations ← [monitor progress, consistent study habits, weak subjects]
    ELSE:
        risk ← "Low Risk"
        recommendations ← [maintain habits, consider advanced challenges]

    RETURN round(score, 2), risk, recommendations
END
```

Model loading uses the same multi-candidate-path search pattern as data loading,
falling back through likely relative locations before raising an error.

---

## 5. Analysis Algorithm
**Purpose:** Compute dashboard-level KPIs, correlations, and risk-level distribution.

```
ALGORITHM GetKPIs(df)
    RETURN {
        Total Students      : count(df),
        Average Score       : mean(Exam_Score),
        Highest Score       : max(Exam_Score),
        Lowest Score        : min(Exam_Score),
        Average Attendance  : mean(Attendance_Rate),
        At-Risk Students    : count(Exam_Score < 60)
    }
END

ALGORITHM GetCorrelationMatrix(df)
    RETURN Pearson correlation matrix over all numeric columns
END

ALGORITHM GetRiskDistribution(df)
    FOR each row:
        Risk_Level ← "High Risk"   if Exam_Score < 60
                    ← "Medium Risk" if 60 <= Exam_Score < 75
                    ← "Low Risk"    if Exam_Score >= 75
    RETURN value_counts(Risk_Level)
END
```

---

## End-to-End Pipeline Summary

```
1. DATA        → LoadOrGenerateData()            → student_performance.csv
2. PREPROCESS  → SplitAndPreprocess()            → train/test sets + preprocessor
3. TRAIN       → TrainAndEvaluate()              → best model (Random Forest) saved as .pkl
4. PREDICT     → PredictPerformance()            → score, risk category, recommendations
5. ANALYZE     → GetKPIs / GetCorrelationMatrix / GetRiskDistribution → dashboard insights
```

**Key design choices:**
- Random Forest is selected as the production model regardless of comparative
  metrics (fixed choice in code), though Linear Regression and Gradient Boosting
  are trained and evaluated alongside it for comparison.
- Risk thresholds (60 / 75) are used consistently across analysis and prediction
  to ensure dashboard and per-student predictions align.
- Missing values are handled via median (numeric) / most-frequent (categorical)
  imputation inside the pipeline, so raw missing data does not need pre-cleaning.
