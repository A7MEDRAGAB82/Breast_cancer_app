import streamlit as st
from data_loader import load_data
from model import get_models, train_and_evaluate, get_feature_importance
from utils import plot_prediction_bar, plot_feature_importance, show_confusion_matrices, local_css

# Load CSS
local_css("style.css")

# Title
st.title("🔬 Breast Cancer Diagnosis")

# Load data
df, X_train, X_test, y_train, y_test, features = load_data()

# Get models
models = get_models()

# Train and evaluate
results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

# Show prediction results
plot_prediction_bar(results)

# Show confusion matrices
show_confusion_matrices(results)

# Feature importance
selected_model_name = st.selectbox("Select model for feature importance", list(results.keys()))
model = results[selected_model_name]["model"]
importances = get_feature_importance(model, X_test, y_test, features)
plot_feature_importance(importances, features)

