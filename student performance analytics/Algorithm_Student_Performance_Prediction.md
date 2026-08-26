# Algorithm: Student Performance Prediction System

## Overview
This system predicts a student's final exam score from academic, behavioral, and
demographic features, classifies the student into a risk category, and generates
supporting analytics. It consists of five stages: **Data Generation/Loading**,
**Preprocessing**, **Model Training**, **Prediction**, and **Analysis**.

The system is built around a classic supervised-learning regression workflow.
Raw tabular data (real or synthetically generated) is cleaned and transformed
into a numeric feature matrix, a regression model learns the relationship
between student attributes and final exam performance, and the trained model
is then reused at inference time to score new, unseen students. A thin
analytics layer sits on top of the raw data to power a reporting/dashboard
view (KPIs, correlations, and risk-band counts) independent of the ML model.

### Why this architecture?
- **Separation of concerns**: data generation, preprocessing, training,
  prediction, and analysis each live in their own module. This makes the
  system easier to test, debug, and extend (e.g., swapping in a new model
  only touches `model_training.py`).
- **Pipeline-based preprocessing**: by bundling the `ColumnTransformer`
  together with the estimator inside a single scikit-learn `Pipeline`, the
  exact same imputation/scaling/encoding logic that was fit on the training
  data is automatically and correctly re-applied at prediction time — this
  avoids a common source of production bugs (train/serve skew).
- **Model comparison before selection**: rather than committing to a single
  algorithm up front, three models spanning different bias/variance
  trade-offs (linear, bagging, boosting) are trained and benchmarked on the
  same split, so the choice of final model is evidence-based rather than
  arbitrary.

### Data Dictionary (features used throughout the pipeline)

| Feature | Type | Description |
|---|---|---|
| Student_ID | identifier | Dropped before modeling |
| Age | numeric | Student age (15–21) |
| Gender | categorical | Male / Female / Other |
| Study_Hours_per_Week | numeric | Weekly self-study hours |
| Attendance_Rate | numeric | % of classes attended |
| Previous_Score | numeric | Prior exam performance |
| Assignment_Performance | numeric | Average assignment score |
| Sleep_Hours_per_Night | numeric | Average nightly sleep |
| Motivation_Level | categorical | Low / Medium / High |
| Parental_Involvement | categorical | Low / Medium / High |
| Internet_Access | categorical | Yes / No |
| Tutoring | categorical | Yes / No |
| Extracurricular_Activities | categorical | Yes / No |
| Family_Income | categorical | Low / Medium / High |
| Distance_from_School | categorical | Near / Moderate / Far |
| Learning_Difficulties | categorical | Yes / No |
| Academic_Support | categorical | Yes / No |
| **Exam_Score** | numeric (target) | Final exam score, 0–100 |

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

### Design Rationale
The synthetic generator does not assign `Exam_Score` randomly; it builds it as
a **structural equation** so the resulting dataset has realistic, learnable
signal for a downstream model to discover — this is essential for a portfolio
or teaching dataset, where the model needs genuine patterns to find rather
than pure noise. Concretely:

- **Linear base term** — 40% weight on Previous_Score reflects the intuition
  that past performance is the single strongest predictor of future
  performance, a well-established finding in educational data mining.
- **Attendance and study time** contribute smaller, additive amounts,
  modeling the idea that effort and presence matter but are secondary to
  established academic trajectory.
- **Non-linear sleep penalty** (`-1.5 * |Sleep_Hours - 7.5|`) intentionally
  introduces a non-monotonic relationship — too little *or* too much sleep
  both hurt performance — which is a useful stress-test for models that
  assume purely linear relationships (this is one reason a plain Linear
  Regression model underperforms tree-based models in evaluation).
- **Interaction effect** between `Learning_Difficulties` and
  `Academic_Support` (a partial offsetting bonus) simulates a realistic
  real-world interaction: support programs mitigate, but do not fully erase,
  the effect of a learning difficulty. Interaction effects like this are only
  recoverable by models capable of learning feature interactions (e.g., tree
  ensembles), again motivating the model comparison step later.
- **Missingness** is injected completely at random (MCAR) in three columns to
  force the pipeline to demonstrate real-world data hygiene (imputation)
  rather than assuming a pristine input file.

### Complexity
Generating `n` students is **O(n)** in both time and memory, since every
column is produced via vectorized NumPy operations across the full array of
students at once (no per-row Python loops). For n = 5000 this completes in a
fraction of a second on commodity hardware.

### Robustness / Path Resolution
`load_or_generate_data` performs a **cascading file search** across four
candidate paths (the given path, and three parent-relative fallbacks) before
falling back to generation. This makes the pipeline resilient to being
invoked from different working directories (e.g., project root vs. `src/`
directory vs. a notebook), which is a common source of `FileNotFoundError`
bugs in multi-module ML codebases.

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

### Why fit-then-transform matters here
The `ColumnTransformer` is deliberately returned **unfit** from
`SplitAndPreprocess` and only fit later, inside the model `Pipeline`, on
`X_train` alone. This ordering matters: if imputation statistics (the median
or the most-frequent category) or scaling statistics (mean/standard
deviation) were computed on the full dataset *before* the train/test split,
information from the test set would leak into training, producing an
optimistically biased evaluation. Fitting exclusively on the training fold
and then simply *applying* the same transform to the test fold is the
correct, leakage-free approach — and wrapping it in a `Pipeline` also
guarantees new incoming records at prediction time are transformed
identically.

### Handling of unseen categories
`OneHotEncoder(handle_unknown='ignore')` ensures that if a prediction-time
request contains a categorical value never seen during training (e.g., a new
`Distance_from_School` label), the encoder emits an all-zero vector for that
feature rather than raising an exception — trading a small amount of
information loss for production robustness.

### Complexity
- `PrepareData`: O(n·m) to scan n rows and m columns for dtype identification.
- `train_test_split`: O(n) via random shuffling and partitioning.
- Fitting `StandardScaler`/`SimpleImputer`: O(n) per numeric column.
- Fitting `OneHotEncoder`: O(n·k) where k is the number of distinct
  categories across categorical columns.
- Overall preprocessing fit cost is **O(n·m)**, linear in dataset size —
  negligible compared to the downstream model-fitting cost.

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

### Model Comparison Rationale
Three fundamentally different learning strategies are benchmarked side by
side so the strengths and weaknesses of each are visible in the results
table rather than assumed:

| Model | Strategy | Strength | Weakness |
|---|---|---|---|
| Linear Regression | Fits a single global linear equation | Fast, interpretable coefficients | Cannot capture the non-linear sleep effect or feature interactions built into the data |
| Random Forest | Bagged ensemble of decorrelated decision trees | Captures non-linearities & interactions, robust to outliers and irrelevant features | Larger model size, less directly interpretable |
| Gradient Boosting | Sequential ensemble that corrects prior trees' residual errors | Often highest raw accuracy on tabular data | More sensitive to hyperparameters, slower to train, can overfit if unconstrained |

Because the synthetic data was engineered with a non-linear sleep-penalty
term and an interaction between `Learning_Difficulties` and
`Academic_Support`, tree-based ensembles (Random Forest, Gradient Boosting)
are expected to outperform plain Linear Regression in the R² comparison —
which the evaluation step confirms empirically rather than by assumption.

### Evaluation Metrics Explained
- **MAE (Mean Absolute Error):** average absolute difference between
  predicted and actual exam scores, in the same units as the score itself
  (points) — easy to communicate to non-technical stakeholders.
- **RMSE (Root Mean Squared Error):** similar to MAE but penalizes large
  errors more heavily due to the squaring term, useful for flagging models
  that occasionally make big mistakes even if average error looks fine.
- **R² (Coefficient of Determination):** proportion of variance in
  Exam_Score explained by the model; 1.0 is a perfect fit, 0.0 means the
  model performs no better than predicting the mean for every student.

### Why Random Forest Is Chosen as the Production Model
The code fixes Random Forest as the final saved model regardless of the
comparison outcome. This is a deliberate simplification for a teaching/demo
project — Random Forest offers a strong, stable balance of accuracy,
resistance to overfitting (via bagging and feature-subsampling), and
out-of-the-box interpretability (via `feature_importances_`), without the
extensive hyperparameter tuning that Gradient Boosting typically needs to
reach its full potential. In a production setting, this choice would
normally be made dynamically by selecting `results_df.iloc[0]` (the top R²
scorer) rather than being hardcoded.

### Complexity
Random Forest training complexity is approximately **O(t · n log(n) · f)**,
where `t` is the number of trees (100), `n` is the number of training
samples, and `f` is the number of features considered per split — this is
higher than Linear Regression's **O(n·f² + f³)** closed-form solution, but
remains tractable for datasets in the thousands-to-low-millions of rows
range, as used here (n ≈ 4000 training rows after an 80/20 split).

### Persistence
The final pipeline (preprocessing + trained model bundled together) is
serialized with `joblib.dump(..., compress=3)`. Saving the *entire pipeline*
— not just the raw estimator — is what allows `prediction.py` to feed raw,
untransformed feature dictionaries directly into `model.predict()` later
without reimplementing any preprocessing logic at inference time.

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

### Worked Example
Given the following input record:

| Feature | Value |
|---|---|
| Study_Hours_per_Week | 22 |
| Attendance_Rate | 91 |
| Previous_Score | 68 |
| Assignment_Performance | 74 |
| Sleep_Hours_per_Night | 7.2 |
| Motivation_Level | High |
| Tutoring | Yes |
| Family_Income | Medium |
| Learning_Difficulties | No |

The pipeline internally: (1) imputes any missing fields using the training
set's medians/most-frequent values — none are missing here — (2) scales the
numeric fields with the fitted `StandardScaler`, (3) one-hot encodes the
categorical fields, then (4) feeds the resulting numeric vector into the
Random Forest regressor. Suppose the model outputs `score = 78.4`. Since
`78.4 ≥ 75`, the algorithm returns:
- **Predicted score:** 78.4
- **Risk category:** Low Risk
- **Recommendations:** maintain current study habits and attendance;
  consider advanced or extracurricular challenges to sustain motivation.

### Feature Importance Extraction
`extract_feature_importance` mirrors the logic used at training time to
recover human-readable feature names after one-hot encoding (which expands a
single categorical column like `Family_Income` into multiple binary columns
such as `Family_Income_Low`, `Family_Income_Medium`, `Family_Income_High`).
Reusing this exact logic at prediction time — rather than only at training
time — allows an application layer (e.g., a dashboard) to explain *why* a
given prediction was risky by showing which features the forest weighted
most heavily overall.

### Error Handling
Both `load_model` and `extract_feature_importance` are defensive: the former
raises a clear `FileNotFoundError` with the attempted path if no model
artifact is found (guiding the user to run training first), and the latter
wraps its extraction logic in a try/except so a malformed or incompatible
pipeline degrades gracefully (returning an empty DataFrame) instead of
crashing the calling application.

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

### Purpose in the System
These three functions power a summary dashboard that is intentionally
**decoupled from the ML model** — they operate directly on the raw/loaded
DataFrame, not on model predictions. This means the analytics view (e.g.,
"how many students are currently at risk based on their actual last exam")
remains available and meaningful even before a model has been trained,
and can be recomputed instantly as new student records are added.

- `GetKPIs` gives a single-glance summary card: cohort size, score spread,
  average attendance, and a live count of currently at-risk students.
- `GetCorrelationMatrix` surfaces linear relationships between numeric
  features (e.g., how strongly `Study_Hours_per_Week` correlates with
  `Exam_Score`) — useful for a heatmap visualization and for sanity-checking
  that the dataset's built-in correlations (from the generation step) are
  visible and sensible.
- `GetRiskDistribution` reuses the identical threshold boundaries (60 / 75)
  that `PredictPerformance` uses for new predictions, so historical risk
  counts and predicted risk counts are always directly comparable —
  avoiding a mismatch where, say, the dashboard calls a 62 "Medium Risk" but
  the prediction endpoint calls it something else.

### Complexity
- `GetKPIs`: O(n) — a small constant number of aggregate passes over n rows.
- `GetCorrelationMatrix`: O(n·m²) where m is the number of numeric columns,
  since a full pairwise correlation matrix is computed.
- `GetRiskDistribution`: O(n) — one categorization pass plus a value count.

All three are cheap enough to be recomputed on every dashboard page load for
datasets in the thousands-of-rows range used by this project.

---

## 6. Application/Dashboard Algorithm (`app.py`)
**Purpose:** Serve the entire system as an interactive multi-page Streamlit
web application, wiring the data, analysis, and prediction modules into a
single user-facing tool.

```
ALGORITHM RunDashboard()
    Configure page (title, icon, wide layout) and inject custom CSS for KPI cards

    df ← cached(LoadOrGenerateData("data/student_performance.csv"))
    TRY:
        model ← LoadModel("models/student_performance_model.pkl")
        model_loaded ← True
    CATCH FileNotFoundError:
        model_loaded ← False

    page ← sidebar radio selection from:
        [Overview, Student Analysis, Performance Analysis,
         At-Risk Student Analysis, Performance Prediction]

    SWITCH page:
        CASE "Overview":          RenderOverviewPage(df, model, model_loaded)
        CASE "Student Analysis":  RenderStudentAnalysisPage(df)
        CASE "Performance Analysis": RenderPerformanceAnalysisPage(df)
        CASE "At-Risk Student Analysis": RenderAtRiskPage(df)
        CASE "Performance Prediction": RenderPredictionPage(model, model_loaded)
END
```

### 6.1 Caching Strategy
`load_data()` is wrapped in `@st.cache_data`, so the (potentially expensive)
data loading/generation step runs only once per session rather than on every
widget interaction — Streamlit reruns the entire script top-to-bottom on
every user action (slider drag, button click, page switch), so caching the
data load is essential to keep the app responsive. The model is *not*
decorated with `@st.cache_resource` in the current code, meaning
`load_model()` re-executes its candidate-path search on every rerun; this is
cheap (a few `os.path.exists` checks plus one `joblib.load` the first time
it succeeds) but would be a natural optimization target for a larger model
file.

### 6.2 Page 1 — Overview
```
ALGORITHM RenderOverviewPage(df, model, model_loaded)
    kpis ← GetKPIs(df)
    Render 6 KPI cards in a 3-column grid (Total Students, Avg/Min/Max Score,
        Avg Attendance, At-Risk Count) using styled HTML cards

    Render histogram of Exam_Score (distribution shape + KDE overlay)
    Render boxplot of Exam_Score grouped by Motivation_Level (Low/Medium/High)

    IF model_loaded:
        importance_df ← ExtractFeatureImportance(model)
        IF importance_df not empty:
            Render horizontal bar chart of top-5 feature importances
            Display the single top feature as a natural-language "Key Insight"
END
```
This page acts as an **executive summary**: it answers "how is the cohort
doing overall?" in one screen, and — only if a trained model is present —
augments the human-authored insight with a data-driven explanation of what
actually drives scores, sourced directly from the Random Forest's learned
`feature_importances_`.

### 6.3 Page 2 — Student Analysis
```
ALGORITHM RenderStudentAnalysisPage(df)
    Collect filter widgets: Gender, Motivation_Level, Tutoring, Family_Income
        (multiselects) and Attendance_Rate, Study_Hours_per_Week (range sliders)

    filtered_df ← df rows matching ALL filter conditions simultaneously
                  (logical AND across every selected filter)

    Display filtered row count and a preview table (first 50 rows)

    IF filtered_df not empty:
        Render bar chart: average Exam_Score grouped by Gender
        Render KDE plot: Exam_Score distribution split by Tutoring status
END
```
This is a **drill-down / segmentation tool**: rather than fixed aggregate
KPIs, the user can interactively slice the cohort (e.g., "low-income
students who receive tutoring") and immediately see how that segment's score
distribution and gender breakdown differ from the whole population. The
empty-check guards (`if not filtered_df.empty`) prevent chart-rendering
errors when a filter combination matches zero students.

### 6.4 Page 3 — Performance Analysis
```
ALGORITHM RenderPerformanceAnalysisPage(df)
    Render scatter + regression line: Attendance_Rate vs Exam_Score
    Render scatter + regression line: Study_Hours_per_Week vs Exam_Score
    corr_matrix ← GetCorrelationMatrix(df)
    Render annotated heatmap of corr_matrix
    Display a static textual insight about the strongest correlated features
END
```
This page is the **exploratory data analysis (EDA)** view: it visually
substantiates two of the strongest engineered relationships in the synthetic
data (attendance and study hours both feed directly into `Exam_Score`) via
scatter plots with fitted regression lines, then generalizes with a full
correlation heatmap so any other numeric relationship can be inspected too.

### 6.5 Page 4 — At-Risk Student Analysis
```
ALGORITHM RenderAtRiskPage(df)
    risk_counts ← GetRiskDistribution(copy of df)
    Render pie chart of risk_counts (High/Medium/Low, fixed color mapping:
        red/yellow/green)

    df['Risk'] ← "High Risk" if Exam_Score < 60 else "Not High Risk"
    Render 3 side-by-side bar charts comparing High-Risk vs Not-High-Risk
        groups on: Attendance_Rate, Study_Hours_per_Week, Sleep_Hours_per_Night

    Display disclaimer: classification supports early intervention,
        not formal diagnosis

    high_risk_df ← rows where Exam_Score < 60, sorted ascending by score
    Display table of the 10 lowest-scoring at-risk students
END
```
This page operationalizes the risk taxonomy for **intervention planning**:
beyond just counting at-risk students, it contrasts their behavioral profile
(attendance, study time, sleep) against the rest of the cohort side by side,
which visually surfaces *which* behavioral factors most separate at-risk
students from their peers — and surfaces a ranked worklist (lowest scorers
first) for staff to prioritize outreach.

### 6.6 Page 5 — Performance Prediction
```
ALGORITHM RenderPredictionPage(model, model_loaded)
    IF NOT model_loaded:
        Show error instructing user to run model_training.py first
        RETURN

    Render a form with 16 input widgets across 3 columns, covering every
        feature the model expects (age, gender, attendance, study hours,
        previous score, assignment performance, sleep, motivation,
        parental involvement, income, distance, internet, tutoring,
        extracurriculars, learning difficulties, academic support)

    ON form submit:
        input_data ← dict assembled from all 16 widget values
        score, risk, recommendations ← PredictPerformance(model, input_data)

        Display predicted score as a metric
        Display risk badge (colored error/warning/success by risk tier)
        Display bullet list of stage-appropriate recommendations

        // Rule-based explanation layer, independent of the model itself:
        factors ← []
        IF attendance < 75:            factors.append("low attendance" note)
        IF study_hours < 10:           factors.append("below-average study hours" note)
        IF previous_score > 80:        factors.append("strong prior performance" note)
        ELSE IF previous_score < 60:   factors.append("weak prior performance" note)
        IF sleep < 6 OR sleep > 9:     factors.append("sub-optimal sleep" note)
        IF learning_diff = Yes AND support = No:
                                        factors.append("unsupported learning difficulty" note)
        IF factors is empty:           factors ← ["Overall profile is balanced."]

        Display each factor as an info callout
END
```

### 6.7 The Two-Layer Explanation Design
A notable design pattern on this page is that **two independent explanation
mechanisms** are shown together:
1. **Model-driven risk tier and recommendations** — purely a function of the
   single predicted `score` number, via fixed thresholds (60/75) baked into
   `predict_performance`.
2. **Rule-based "major factors" callouts** — a separate, hand-written set of
   `if/elif` checks directly on the *raw input values* the user typed in,
   completely independent of the trained model's internal logic.

This second layer exists because a Random Forest's actual decision path for
a single prediction is not naturally human-readable in real time; the
hand-authored heuristics give users an immediately understandable,
plausible-sounding explanation ("low attendance is hurting your score")
without requiring a full SHAP/LIME-style model-explainability integration.
The trade-off is that these rule-based factors are **not guaranteed to match
the model's actual reasoning** for that specific prediction — they are a
reasonable proxy based on domain knowledge of how the data was generated,
not a faithful decomposition of the model's decision.

### 6.8 State and Control Flow Notes
- Streamlit's execution model reruns the whole script on every interaction,
  so all page logic is written as straight-line procedural code guarded by
  `if/elif` on the sidebar selection — there is no persistent server-side
  session state beyond what `st.cache_data` retains.
- The prediction form uses `st.form(...)` specifically so that none of the
  16 widget changes trigger a rerun/prediction individually — only the
  final "Predict Performance" submit button does, batching all inputs into
  a single `predict_performance` call.
- Graceful degradation: if no trained model exists yet, the Overview page
  simply omits the feature-importance section, and the Prediction page
  replaces its form entirely with an instructional error — the rest of the
  dashboard (Overview KPIs/charts, Student Analysis, Performance Analysis,
  At-Risk Analysis) works from `df` alone and needs no model at all.

---

## End-to-End Pipeline Summary

```
1. DATA        → LoadOrGenerateData()            → student_performance.csv
2. PREPROCESS  → SplitAndPreprocess()            → train/test sets + preprocessor
3. TRAIN       → TrainAndEvaluate()              → best model (Random Forest) saved as .pkl
4. PREDICT     → PredictPerformance()            → score, risk category, recommendations
5. ANALYZE     → GetKPIs / GetCorrelationMatrix / GetRiskDistribution → dashboard insights
6. SERVE       → app.py (Streamlit)              → 5-page interactive dashboard tying 1–5 together
```

**Key design choices:**
- Random Forest is selected as the production model regardless of comparative
  metrics (fixed choice in code), though Linear Regression and Gradient Boosting
  are trained and evaluated alongside it for comparison.
- Risk thresholds (60 / 75) are used consistently across analysis, prediction,
  and the dashboard's At-Risk page to ensure they always align.
- Missing values are handled via median (numeric) / most-frequent (categorical)
  imputation inside the pipeline, so raw missing data does not need pre-cleaning.
- The dashboard is designed to degrade gracefully when no trained model is
  present, keeping all data-driven (non-ML) pages fully functional.
- The prediction page pairs true model output (score/risk/recommendations)
  with an independent, hand-authored rule-based explanation layer, trading
  perfect faithfulness to the model's internals for immediate interpretability.
