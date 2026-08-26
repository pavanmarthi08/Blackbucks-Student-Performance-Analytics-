import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_or_generate_data
from src.analysis import get_kpis, get_correlation_matrix, get_risk_distribution
from src.prediction import load_model, predict_performance, extract_feature_importance

# Set page config
st.set_page_config(page_title="Student Analytics Dashboard", page_icon="🎓", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .kpi-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
    }
    .kpi-title {
        color: #6c757d;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .kpi-value {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    return load_or_generate_data("data/student_performance.csv")

df = load_data()

# Try loading the model
try:
    model = load_model("models/student_performance_model.pkl")
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    
# Sidebar Navigation
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Select a Page", [
    "Overview", 
    "Student Analysis", 
    "Performance Analysis", 
    "At-Risk Student Analysis", 
    "Performance Prediction"
])

# --- PAGE 1: OVERVIEW ---
if page == "Overview":
    st.title("📊 Student Performance Overview")
    st.markdown("A high-level view of student academic success and major factors.")
    
    kpis = get_kpis(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Students</div><div class="kpi-value">{kpis["Total Students"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Lowest Score</div><div class="kpi-value">{kpis["Lowest Score"]}%</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Average Score</div><div class="kpi-value">{kpis["Average Score"]}%</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Average Attendance</div><div class="kpi-value">{kpis["Average Attendance"]}%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Highest Score</div><div class="kpi-value">{kpis["Highest Score"]}%</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">At-Risk Students (<60%)</div><div class="kpi-value" style="color: #e74c3c;">{kpis["At-Risk Students"]}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Exam Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df['Exam_Score'], kde=True, bins=30, color="skyblue", ax=ax)
        ax.set_xlabel("Exam Score")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Performance by Motivation Level")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x='Motivation_Level', y='Exam_Score', data=df, order=["Low", "Medium", "High"], palette="Set2", ax=ax)
        st.pyplot(fig)
        
    if model_loaded:
        st.markdown("### Top Factors Influencing Performance")
        st.info("Insights generated automatically by our Machine Learning model.")
        importance_df = extract_feature_importance(model)
        if not importance_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(5), palette="viridis", ax=ax)
            st.pyplot(fig)
            
            top_feature = importance_df.iloc[0]['Feature']
            st.success(f"**Key Insight:** Our model indicates that '{top_feature}' is the strongest predictor of a student's final exam score.")

# --- PAGE 2: STUDENT ANALYSIS ---
elif page == "Student Analysis":
    st.title("🧑‍🎓 Student Analysis")
    st.markdown("Filter and analyze specific student segments.")
    
    with st.expander("Filter Students", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender_filter = st.multiselect("Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
            motivation_filter = st.multiselect("Motivation Level", options=df['Motivation_Level'].unique(), default=df['Motivation_Level'].unique())
        with col2:
            attendance_range = st.slider("Attendance Rate (%)", min_value=float(df['Attendance_Rate'].min()), max_value=float(df['Attendance_Rate'].max()), value=(50.0, 100.0))
            study_hours = st.slider("Study Hours (Weekly)", min_value=float(df['Study_Hours_per_Week'].min()), max_value=float(df['Study_Hours_per_Week'].max()), value=(0.0, 40.0))
        with col3:
            tutoring_filter = st.multiselect("Tutoring", options=df['Tutoring'].unique(), default=df['Tutoring'].unique())
            income_filter = st.multiselect("Family Income", options=df['Family_Income'].unique(), default=df['Family_Income'].unique())

    # Apply filters
    filtered_df = df[
        (df['Gender'].isin(gender_filter)) &
        (df['Motivation_Level'].isin(motivation_filter)) &
        (df['Tutoring'].isin(tutoring_filter)) &
        (df['Family_Income'].isin(income_filter)) &
        (df['Attendance_Rate'] >= attendance_range[0]) & (df['Attendance_Rate'] <= attendance_range[1]) &
        (df['Study_Hours_per_Week'] >= study_hours[0]) & (df['Study_Hours_per_Week'] <= study_hours[1])
    ]
    
    st.markdown(f"**Showing {len(filtered_df)} students based on filters.**")
    st.dataframe(filtered_df[['Student_ID', 'Age', 'Gender', 'Attendance_Rate', 'Study_Hours_per_Week', 'Motivation_Level', 'Exam_Score']].head(50))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Score by Gender")
        if not filtered_df.empty:
            avg_by_gender = filtered_df.groupby("Gender")["Exam_Score"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x="Gender", y="Exam_Score", data=avg_by_gender, palette="pastel", ax=ax)
            ax.set_ylim(0, 100)
            st.pyplot(fig)
            
    with col2:
        st.subheader("Tutoring Impact")
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.kdeplot(data=filtered_df, x="Exam_Score", hue="Tutoring", fill=True, palette="muted", ax=ax)
            st.pyplot(fig)


# --- PAGE 3: PERFORMANCE ANALYSIS ---
elif page == "Performance Analysis":
    st.title("📈 Performance Analysis")
    st.markdown("Deep dive into the relationships between behaviors and academic success.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attendance vs Exam Score")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x="Attendance_Rate", y="Exam_Score", data=df, alpha=0.5, color="coral", ax=ax)
        sns.regplot(x="Attendance_Rate", y="Exam_Score", data=df, scatter=False, color="red", ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Study Hours vs Exam Score")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x="Study_Hours_per_Week", y="Exam_Score", data=df, alpha=0.5, color="teal", ax=ax)
        sns.regplot(x="Study_Hours_per_Week", y="Exam_Score", data=df, scatter=False, color="darkblue", ax=ax)
        st.pyplot(fig)
        
    st.markdown("---")
    st.subheader("Correlation Heatmap")
    corr_matrix = get_correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    st.success("**Insight:** Notice the strong positive correlations between Previous_Score, Attendance_Rate, Study_Hours_per_Week, and the final Exam_Score. These are clear indicators of academic success.")

# --- PAGE 4: AT-RISK ANALYSIS ---
elif page == "At-Risk Student Analysis":
    st.title("⚠️ At-Risk Student Analysis")
    st.markdown("Identify students who are potentially academically at risk based on the standard thresholds (High Risk: < 60%, Medium Risk: 60-75%, Low Risk: > 75%).")
    
    risk_counts = get_risk_distribution(df.copy())
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots(figsize=(5,5))
        colors = ['#e74c3c', '#f1c40f', '#2ecc71']
        labels = ['High Risk', 'Medium Risk', 'Low Risk']
        # Ensure ordering matches
        sizes = [risk_counts.get('High Risk', 0), risk_counts.get('Medium Risk', 0), risk_counts.get('Low Risk', 0)]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Key Factors for High-Risk Students")
        df['Risk'] = np.where(df['Exam_Score'] < 60, 'High Risk', 'Not High Risk')
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        
        sns.barplot(x="Risk", y="Attendance_Rate", data=df, ax=ax[0], palette="Set1")
        ax[0].set_title("Avg Attendance")
        
        sns.barplot(x="Risk", y="Study_Hours_per_Week", data=df, ax=ax[1], palette="Set1")
        ax[1].set_title("Avg Study Hours")
        
        sns.barplot(x="Risk", y="Sleep_Hours_per_Night", data=df, ax=ax[2], palette="Set1")
        ax[2].set_title("Avg Sleep Hours")
        
        st.pyplot(fig)
        
    st.warning("**Note:** This classification is intended to provide early academic support recommendations, not to formally diagnose a student's capabilities.")
    
    st.subheader("Students Requiring Attention (Sample)")
    high_risk_df = df[df['Exam_Score'] < 60].sort_values(by='Exam_Score')
    st.dataframe(high_risk_df[['Student_ID', 'Exam_Score', 'Attendance_Rate', 'Study_Hours_per_Week', 'Motivation_Level']].head(10))

# --- PAGE 5: PREDICTION ---
elif page == "Performance Prediction":
    st.title("🔮 Student Performance Prediction")
    st.markdown("Use our Machine Learning model to predict a student's final exam score based on their profile.")
    
    if not model_loaded:
        st.error("Model not found! Please run the training script (`python src/model_training.py`) to generate the model before using this page.")
    else:
        st.markdown("### Enter Student Details")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=15, max_value=30, value=18)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
                study_hours = st.slider("Study Hours (Weekly)", 0, 50, 15)
                previous_score = st.slider("Previous Score (%)", 0, 100, 75)
                assignment = st.slider("Assignment Performance (%)", 0, 100, 75)
                
            with col2:
                sleep = st.slider("Sleep Hours per Night", 2.0, 12.0, 7.5, 0.5)
                motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
                parental = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
                income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
                distance = st.selectbox("Distance from School", ["Near", "Moderate", "Far"])
                
            with col3:
                internet = st.selectbox("Internet Access", ["Yes", "No"])
                tutoring = st.selectbox("Tutoring", ["Yes", "No"])
                extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
                learning_diff = st.selectbox("Learning Difficulties", ["Yes", "No"], index=1)
                support = st.selectbox("Academic Support", ["Yes", "No"], index=1)
                
            submit = st.form_submit_button("Predict Performance")
            
        if submit:
            input_data = {
                'Age': age,
                'Gender': gender,
                'Attendance_Rate': attendance,
                'Study_Hours_per_Week': study_hours,
                'Previous_Score': previous_score,
                'Assignment_Performance': assignment,
                'Sleep_Hours_per_Night': sleep,
                'Motivation_Level': motivation,
                'Parental_Involvement': parental,
                'Family_Income': income,
                'Distance_from_School': distance,
                'Internet_Access': internet,
                'Tutoring': tutoring,
                'Extracurricular_Activities': extra,
                'Learning_Difficulties': learning_diff,
                'Academic_Support': support
            }
            
            score, risk, recommendations = predict_performance(model, input_data)
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Predicted Exam Score", f"{score}%")
                
                if risk == "High Risk":
                    st.error(f"**Risk Level:** {risk}")
                elif risk == "Medium Risk":
                    st.warning(f"**Risk Level:** {risk}")
                else:
                    st.success(f"**Risk Level:** {risk}")
                    
            with res_col2:
                st.markdown("#### Recommendations:")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                    
            # Explanatory insights generated dynamically based on model rules
            st.markdown("#### Major factors influencing this prediction:")
            factors = []
            if attendance < 75: factors.append("Low attendance is heavily impacting the score negatively.")
            if study_hours < 10: factors.append("Study hours are below average.")
            if previous_score > 80: factors.append("Strong previous academic performance is a major positive indicator.")
            elif previous_score < 60: factors.append("Poor previous scores strongly suggest need for foundational review.")
            if sleep < 6 or sleep > 9: factors.append("Sub-optimal sleep schedule may be affecting cognitive performance.")
            if learning_diff == 'Yes' and support == 'No': factors.append("Unaddressed learning difficulties represent a significant hurdle. Academic support is strongly advised.")
            
            if not factors:
                factors.append("Overall profile is balanced.")
                
            for f in factors:
                st.info(f)
