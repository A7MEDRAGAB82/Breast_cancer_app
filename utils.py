import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def local_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def plot_prediction_result(pred, proba):
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    if pred == 1:
        st.markdown('<p class="benign" style="font-size: 2rem;">BENIGN</p>', unsafe_allow_html=True)
        st.success("Tumor is likely non-cancerous.")
    else:
        st.markdown('<p class="malignant" style="font-size: 2rem;">MALIGNANT</p>', unsafe_allow_html=True)
        st.error("Tumor is likely cancerous.")
    st.write(f"Confidence: {max(proba):.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots()
    ax.barh(['Benign', 'Malignant'], proba, color=['#10b981', '#ef4444'])
    ax.set_xlim(0, 1)
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

def plot_feature_importance(features, importances):
    df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Importance', y='Feature', ax=ax, palette='viridis')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

def show_dataset_overview(df):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.write(f"Samples: {df.shape[0]} | Features: {df.shape[1] - 2}")
        st.write(f"Malignant: {len(df[df['target'] == 0])}")
        st.write(f"Benign: {len(df[df['target'] == 1])}")
        st.markdown('</div>', unsafe_allow_html=True)

def plot_target_distribution(df):
    fig, ax = plt.subplots()
    df['diagnosis'].value_counts().plot.pie(
        autopct='%1.1f%%', colors=['#ef4444', '#10b981'], ax=ax
    )
    ax.set_ylabel('')
    st.pyplot(fig)

def plot_feature_distribution(df, features):
    selected = st.selectbox("Select feature", features)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='diagnosis', y=selected,
                palette={'Malignant': '#ef4444', 'Benign': '#10b981'})
    ax.set_title(f'{selected} by Diagnosis')
    st.pyplot(fig)

def show_model_comparison(results, best_name):
    df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[k]['accuracy'] for k in results],
        'Selected': [k == best_name for k in results]
    }).sort_values('Accuracy', ascending=False)
    st.dataframe(df.style.format({'Accuracy': '{:.2%}'}), use_container_width=True)

def show_confusion_matrices(results):
    cols = st.columns(len(results))
    for idx, (name, res) in enumerate(results.items()):
        with cols[idx]:
            st.write(f"**{name}**")
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.heatmap(res['confusion'], annot=True, fmt='d',
                        cmap='Blues', ax=ax,
                        xticklabels=['Malignant', 'Benign'],
                        yticklabels=['Malignant', 'Benign'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
