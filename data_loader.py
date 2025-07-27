import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
    return df, data.feature_names.tolist()
