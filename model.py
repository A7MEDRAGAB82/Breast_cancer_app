from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

def get_models():
    return {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(probability=True)
    }

def train_and_evaluate(models, X, y):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(
        StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42
    )
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion": confusion_matrix(y_test, y_pred)
        }
    return results

def get_best_model(results):
    best_name = max(results, key=lambda x: results[x]['accuracy'])
    return best_name, results[best_name]['model']

def get_feature_importance(model, X, y, features):
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    else:
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        return result.importances_mean
