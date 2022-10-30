import joblib
import numpy as np
from src.explainable_ai.xai_feature_imporance_metrics import XaiFeatureImportanceMetrics
from lime.lime_tabular import LimeTabularExplainer
import shap
import os


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
    monotonocity_dict = {"monotonocity": monotonous_array}
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
    faith_ful_dict = {'Faithfulness': fait_fulness_array}
    return fait_fulness_array


def xai_metrics_scores(X_test, features, target_names, models_res_path):
    """

    :param X_test: (pandas.core.frame.DataFrame) data
    :param features: (list) features
    :param target_names: (list)
    :param models_res_path: folder location that contain trained model names
    """
    model_strings = ['KNeighborsClassifier',
                     'ExtraTreesClassifier',
                     'LGBMClassifier',
                     'RandomForestClassifier',
                     'GradientBoostingClassifier',
                     'XGBClassifier',
                     'DecisionTreeClassifier']
    print("==" * 40)
    # X_test = X_test[features]
    results_dict = {}
    monotonocity = {}
    faithfulness = {}
    files = os.listdir(models_res_path)  # Folder name of trained models
    for string_name in model_strings:
        file_name = [s for s in files if string_name in s][0]
        file_location = f'{models_res_path}/{file_name}'
        print("file Location:{0}\n".format(file_location))
        print("model:{0}\n".format(string_name))
        model_name = joblib.load(file_location)
        model = model_name.best_estimator_
        faith_fulness_score = lime_faith_fulness(X_test, features, model, target_names)
        faithfulness['faithfulness'] = faith_fulness_score
        monotonocity_score = lime_monotonicity(X_test, features, model, target_names)
        monotonocity['monotonocity'] = monotonocity_score
        results_dict[string_name] = [faithfulness, monotonocity]
        monotonic_mean = np.array(monotonocity_score)
        with open(f'{models_res_path}/xai_metrics_res.txt', 'a') as res_logs:
            res_logs.write("==" * 40)
            res_logs.write("Model Name:{0}\n".format(string_name))
            res_logs.write("Faithfulness Metric mean :{0}\n".format(str(np.mean(faith_fulness_score))))
            res_logs.write("Faithfulness metric std:{0}\n".format(str(np.std(faith_fulness_score))))
            res_logs.write("{0}% of Record where Explanation is monotonic\n".format(monotonic_mean))
            res_logs.write("==" * 40)
    file_name = f'{models_res_path}/xai_models_res.pkl'
    print("file name:{0}\n".format(file_name))
    joblib.dump(results_dict, file_name)
    return results_dict


