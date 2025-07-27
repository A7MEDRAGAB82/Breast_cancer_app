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

def show_confusion_matrices(results):
    cols = st.columns(len(results))
    for idx, (name, result) in enumerate(results.items()):
        with cols[idx]:
            st.write(f"**{name}**")
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.heatmap(result['confusion'], annot=True, fmt='d',
                        cmap='Blues', ax=ax,
                        xticklabels=['Malignant', 'Benign'],
                        yticklabels=['Malignant', 'Benign'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

def local_css(style):
    with open(style) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
