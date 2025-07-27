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