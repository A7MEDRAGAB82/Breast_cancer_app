from data_loader import load_data
from model import get_models, train_and_evaluate, get_feature_importance
from utils import plot_prediction_bar, plot_feature_importance, show_confusion_matrices

df = load_data()

models = get_models()
results = train_and_evaluate(models, df)
