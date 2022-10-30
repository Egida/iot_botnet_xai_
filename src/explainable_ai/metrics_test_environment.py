import numpy as np
from src.explainable_ai.xai_feature_imporance_metrics import XaiFeatureImportanceMetrics
import lime
import shap


def lime_monotonicity(X_test, model):
    """
    :param X_test: (pandas.core.frame.DataFrame)
    :param model: scikit-learn model , or pre-trained model.
    """
    # montonocity metric
    print("==" * 40)
    print("\n")
    print("Lime Monotonicity")
    number_of_features = X_test.shape[1]
    X = X_test.to_numpy()
    XaiFeatureImportanceMetrics().monotonicity_metric(model, X, )
    return 0
