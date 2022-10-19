from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics
import numpy as np
import pandas as pandas


class parameter_tuning:
    """
    Tuning Algorithms
    """

    def __int__(self, X, y, metric_type, cv):
        """

        :param X: Independent variable
        :param y: Dependent Variable
        :param metric_type: 1. accuracy, f1-score, recall, precision
        :param cv: Cross validation score
        :return:
        """
        self.X = X,
        self.y = y,
        self.metric_type = metric_type
        self.cv = cv

    def _fit_grid_random_search(self, X, y, ml_classifier, parameters, search_type):

        """ Training the model using Grid search or Random search hyperparameter tuning methods.

        :param X: Independent Variable
        :param y: Dependent Variable
        :param ml_classifier: Scikit learn classifier
        :param parameters: various combinations for parameters for classifier.
        :param search_type: Search type are 1. grid_search 2. random_search
        :return:
        """
        # Classifier name
        mlclassifier_name = str(type(ml_classifier)).split(".")[-1][:-2]
        print("Classifier is {0}".format(mlclassifier_name))
        # check the Parameter type,
        if search_type == 'grid_search':
            # Grid Search parameter type
            tuned_model = GridSearchCV(ml_classifier,
                                       param_grid=parameters,
                                       scoring=self.metric_type,
                                       verbose=10,
                                       cv=self.cv)
        elif search_type == 'random_search':
            # Random Search Parameter Tuning
            tuned_model = RandomizedSearchCV(estimator=ml_classifier,
                                             param_distributions=parameters,
                                             scoring=self.metric_type,
                                             cv=self.cv,
                                             verbose=10)

        else:
            print("===========================================")
            print(f'{search_type} is wrong key word.'
                  f'Key word should be either 1.grid_search or 2.random_search')

        tuned_model = tuned_model.fit(self.X, self.y)

        with open('reports/parameter_tuning.txt', 'a') as res_logs:
            res_logs.write('hello world')
            res_logs.write('==' * 40)
            res_logs.write("\n")
            res_logs.write("1.Classifier:{0}\n".format(mlclassifier_name))
            res_logs.write("2.Best Parameters:{0}\n".format(str(tuned_model.best_params_)))

        return 0


# %%
from sklearn.linear_model import LogisticRegression

print(LogisticRegression().__class__.__name__)

# %%
classifier_name = "hellow world"
with open('reports/parameter_tuning.txt', 'a') as results:
    results.write('==' * 40)
    results.write("\n")
    results.write("1.Classifier:{0}\n".format(classifier_name))
