from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

def get_models():
    return {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(probability=True)
    }

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "confusion": confusion_matrix(y_test, y_pred)
        }
    return results

def get_feature_importance(model, X_test, y_test, features):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = result.importances_mean
    return importances