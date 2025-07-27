import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_prediction_bar(proba):
    fig, ax = plt.subplots()
    ax.barh(['Benign', 'Malignant'], proba, color=['#10b981', '#ef4444'])
    ax.set_xlim(0, 1)
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

def plot_feature_importance(features, importances):
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_title('Feature Importance for Prediction')
    st.pyplot(fig)