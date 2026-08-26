# Student Analysis & Prediction Dashboard 🎓

A complete data science and machine learning dashboard built with Python and Streamlit. This project analyzes student academic performance, identifies major factors affecting performance, detects at-risk students, and predicts future exam scores using Machine Learning.

## Features
- **Exploratory Data Analysis (EDA):** Interactive charts showing correlations between study habits, attendance, and exam scores.
- **Machine Learning Pipeline:** Trains multiple regression models (Linear Regression, Random Forest, Gradient Boosting) and automatically selects the best performer.
- **Interactive Dashboard:** A 5-page Streamlit app.
- **At-Risk Prediction:** Classifies students into Low, Medium, and High Risk to offer early academic intervention.
- **Explainable AI:** Highlights the top factors influencing a student's score dynamically.

## Project Structure
```
student_analysis_dashboard/
├── data/                       # Contains dataset (CSV)
├── notebooks/                  # Jupyter notebooks for EDA
├── models/                     # Saved joblib models (.pkl)
├── src/                        # Core Python logic
│   ├── data_processing.py      # Data loading and synthetic data generation
│   ├── preprocessing.py        # ML pipelines, imputers, and scalers
│   ├── analysis.py             # Logic for KPI calculations and metrics
│   ├── model_training.py       # ML training and evaluation script
│   └── prediction.py           # Inference functions
├── dashboard/                  # Streamlit application
│   └── app.py                  
├── requirements.txt
└── README.md
```

## How to Run

1. **Install Requirements:**
```bash
pip install -r requirements.txt
```

2. **Generate Data and Train the ML Model:**
Run the training script to generate the synthetic dataset (or process an existing dataset) and output the trained Random Forest model:
```bash
python src/model_training.py
```
*Note: A highly realistic synthetic dataset is generated automatically if `student_performance.csv` is not found, ensuring the dashboard works out-of-the-box.*

3. **Run the Dashboard:**
```bash
streamlit run dashboard/app.py
```

## Technologies Used
- Python, Pandas, NumPy
- Scikit-learn (Random Forest, Linear Regression, Pipeline, Preprocessing)
- Matplotlib, Seaborn
- Streamlit
