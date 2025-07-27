import pandas as pd
from sklearn.datasets import load_breast_cancer
import streamlit as st

@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
    features = [
        'mean radius', 'mean texture', 'mean perimeter',
        'mean smoothness', 'mean compactness', 'mean concavity'
    ]
    return df, features

def user_input_features(df):
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/Pink_ribbon.svg", width=100)
    st.sidebar.title("Breast Cancer Prediction")
    st.sidebar.markdown("Adjust tumor characteristics to predict malignancy.")

    inputs = {}
    with st.sidebar.expander("Size Features", expanded=True):
        for feature in ['mean radius', 'mean texture', 'mean perimeter']:
            min_val, max_val = float(df[feature].min()), float(df[feature].max())
            inputs[feature] = st.slider(feature, min_val, max_val, float(df[feature].mean()))

    with st.sidebar.expander("Shape Features"):
        for feature in ['mean smoothness', 'mean compactness', 'mean concavity']:
            min_val, max_val = float(df[feature].min()), float(df[feature].max())
            inputs[feature] = st.slider(feature, min_val, max_val, float(df[feature].mean()))

    return pd.DataFrame([inputs])
