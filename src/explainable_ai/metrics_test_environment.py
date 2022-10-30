import numpy as np
from src.explainable_ai.xai_feature_imporance_metrics import XaiFeatureImportanceMetrics
from lime.lime_tabular import LimeTabularExplainer
import shap


def lime_monotonicity(X_test, features, model, target_names):
    """
    :param target_names: (list) of target of
    :param features: features from feature selection
    :param X_test: (pandas.core.frame.DataFrame)
    :param model: scikit-learn model , or pre-trained model.
    """
    # monotonicity metric
    explainer = LimeTabularExplainer(X_test.to_numpy(),
                                     feature_names=features,
                                     class_names=target_names,
                                     discretize_continuous=True)

    metric = XaiFeatureImportanceMetrics()
    print("==" * 40)
    print("\n")
    print("Lime Monotonicity\n")
    n_cases = X_test.shape[0]
    monotonous_array = np.zeros(n_cases)
    X = X_test[features].to_numpy()
    for i in range(n_cases):
        predicted_class = model.predict(X[i].reshape(1, -1))[0]
        exp = explainer.explain_instance(X[i], model.predict_proba, num_features=len(features), top_labels=1)
        local_exp = exp.local_exp[predicted_class]
        m = exp.as_map()
        x_values = X[i]
        base = np.zeros(x_values.shape)
        co_efs = np.zeros(x_values.shape[0])
        for value in local_exp:
            co_efs[value[0]] = value[1]
        monotonous_array[i] = metric.monotonicity_metric(model, X[i], co_efs, base)
    monotonic_mean = np.array(monotonous_array)
    print("{0}% of Record where Explanation is monotonic\n".format(monotonic_mean))
    print("==" * 40)
    return monotonous_array


def lime_faith_fulness(X_test, features, model, target_names):
    """

    :param X_test:(pandas.core.frame.DataFrame) data
    :param features: (list) features
    :param model: trained model scikit learn
    :param target_names:  (List) target names
    """
    metric = XaiFeatureImportanceMetrics()
    print("==" * 40)
    print("\n")
    print("Lime Faith Fulness\n")
    X_test = X_test[features].to_numpy()
    n_cases = X_test.shape[0]
    explainer = LimeTabularExplainer(X_test,
                                     feature_names=features,
                                     class_names=target_names,
                                     discretize_continuous=True)

    fait_fulness_array = np.zeros(n_cases)
    for i in range(n_cases):
        predicted_class = model.predict(X_test[i].reshape(1, -1))[0]
        exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=5, top_labels=1)
        localexp = exp.local_exp[predicted_class]
        m = exp.as_map()
        x_values = X_test[i]
        base = np.zeros(x_values.shape[0])
        co_efs = np.zeros(x_values.shape[0])
        for value in localexp:
            co_efs[value[0]] = value[1]
        fait_fulness_array[i] = metric.faithfulness_metric(model, X_test[i], co_efs, base)
    faith_mean = np.mean(fait_fulness_array)
    faith_std = np.std(fait_fulness_array)
    print("Faithfulness Metric mean :{0}\n".format(faith_mean))
    print("Faithfulness Metric Std :{0}\n".format(faith_std))
    return fait_fulness_array


def xai_metrics_scores(X_test, features, model, target_names, file_location):
    """

    :param X_test: (pandas.core.frame.DataFrame) data
    :param features:
    :param model:
    :param target_names:
    :param file_location:
    """

