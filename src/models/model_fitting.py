from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics
import numpy as np
import pandas as pandas


class tuning_algorithmms:
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

        if search_type == 'grid_search':
            # adding the parameters
            hyper_param_model = GridSearchCV(ml_classifier,
                                             param_grid=parameters,
                                             scoring=self.metric_type,
                                             verbose=10,
                                             cv=self.cv)
            hyper_param_model.fit(self.X, self.y)  # fitting the grid search.
        elif search_type == 'random_search':
            hyper_param_model = RandomizedSearchCV(estimator=ml_classifier,
                                                     param_distributions=parameters,
                                                     scoring=self.metric_type,
                                                     cv=self.cv,
                                                     verbose=4)
            hyper_param_model.fit(self.X, self.y)
        else:
            print("===========================================")
            print(f'{search_type} is wrong key word.'
                  f'Key word should be either 1.grid_search or 2.random_search')



        with open('reports/parameter_tuning', 'a') as results:
            results.write('hello world')

        return 0
