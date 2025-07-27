import streamlit as st
import pandas as pd
from data_loader import load_data, user_input_features
from model import get_models, train_and_evaluate, get_best_model, get_feature_importance
from utils import (
    local_css, plot_prediction_result, plot_feature_importance,
    show_dataset_overview, plot_target_distribution,
    plot_feature_distribution, show_model_comparison,
    show_confusion_matrices
)

st.set_page_config(
    page_title="Breast Cancer Diagnosis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
local_css("style.css")

# Load and process data
df, features = load_data()
user_df = user_input_features(df)

# Prepare training data
X = df[features]
y = df['target']

# Train models
models = get_models()
results = train_and_evaluate(models, X, y)

# Select best model
best_model_name, best_model = get_best_model(results)

# Predict
prediction = best_model.predict(user_df)[0]
proba = best_model.predict_proba(user_df)[0]

# Layout
st.markdown('<p class="header-text">Breast Cancer Malignancy Predictor</p>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Analysis", "Model Details"])

with tab1:
    st.subheader("Diagnosis Prediction")
    plot_prediction_result(prediction, proba)

    st.subheader("Feature Importance")
    importances = get_feature_importance(best_model, X, y, features)
    plot_feature_importance(features, importances)

with tab2:
    st.subheader("Dataset Overview")
    show_dataset_overview(df)

    st.subheader("Target Distribution")
    plot_target_distribution(df)

    st.subheader("Feature Distributions by Diagnosis")
    plot_feature_distribution(df, features)

with tab3:
    st.subheader("Model Performance Comparison")
    show_model_comparison(results, best_model_name)
    st.subheader("Confusion Matrices")
    show_confusion_matrices(results)

    st.subheader("About the Models")
    st.markdown("""
    - **Logistic Regression**: Linear classification model.
    - **Random Forest**: Ensemble of decision trees.
    - **Support Vector Machine**: Margin-based classifier.

    Best model is selected automatically based on accuracy.
    """)

st.markdown("---")


