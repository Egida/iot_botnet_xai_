import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics
import numpy as np
import pandas as pandas
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import randint as sp_randint
from random import randrange as sp_randrange


class parameter_tuning:
    """
    Tuning Algorithms
    """

    def __int__(self, X, y, metric_type, cv, search_type='grid_search'):
        """
        Tuning the algorithms

        :param X: Independent variable
        :param y: Dependent Variable
        :param metric_type: 1. accuracy, f1-score, recall, precision
        :param cv: Cross validation score
        :param search_type: Hyper Parameter Tuning type 1. Grid Search 2. Random Search type
        :return:
        """

        self.X = X
        self.y = y
        self.metric_type = metric_type,
        self.cv = cv,
        self.search_type = search_type

    def _fit_grid_random_search(self,ml_classifier, parameters):

        """ Training the model using Grid search or Random search hyperparameter tuning methods.

        :param X: Independent Variable
        :param y: Dependent Variable
        :param ml_classifier: Scikit learn classifier
        :param parameters: various combinations for parameters for classifier.
        :return:
        """
        # Classifier name
        mlclassifier_name = str(type(ml_classifier)).split(".")[-1][:-2]
        print("Classifier is {0}".format(mlclassifier_name))
        # check the Parameter type,

        if self.search_type == 'grid_search':
            # Grid Search parameter type
            tuned_model = GridSearchCV(ml_classifier,
                                       param_grid=parameters,
                                       scoring=self.metric_type,
                                       verbose=10,
                                       cv=self.cv,
                                       refit=False)
        elif self.search_type == 'random_search':
            # Random Search Parameter Tuning
            tuned_model = RandomizedSearchCV(estimator=ml_classifier,
                                             param_distributions=parameters,
                                             scoring=self.metric_type,
                                             cv=self.cv,
                                             verbose=10,
                                             refit=False)

        else:
            print("===========================================")
            print(f'{search_type} is wrong key word.'
                  f'Key word should be either 1.grid_search or 2.random_search')

        print(tuned_model)
        # Finally fit the Classifier
        start_time = self.timer(0) # starting time for model training
        tuned_model = tuned_model.fit(self.X, self.y) # fitting the model
        finish_time = self.timer(start_time) # Finishing for model training
        # Save the model
        joblib.dump(tuned_model, f'reports/{mlclassifier_name}.pkl')
        with open('reports/parameter_tuning.txt', 'a') as res_logs:
            res_logs.write('hello world')
            res_logs.write('==' * 40)
            res_logs.write("\n")
            res_logs.write("1.Classifier:{0}\n".format(mlclassifier_name))
            res_logs.write("2.Best Parameters:{0}\n".format(str(tuned_model.best_params_)))
            res_logs.write("3.Finishing time")

        return 0

    # Time to  count the model for training.
    def timer(self, start_time=None):
        """
        :param start_time: 0
        :return: Completion time
        """
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            # print("\n Time taken: %i hours %i minutes and %s seconds" % (thour, tmin, round(tsec,2)))
        return str("Time consumption: %i hours %i minutes and %s seconds" % (thour, tmin, round(tsec, 2)))

    def rf_classification(self):
        """
        Random Forest Classifier
        """
        # Initiate the classifier
        classifier = RandomForestClassifier(n_jobs=-1)

        # parameters for grid search
        rf_params = {
            'max_features': ['sqrt', 'auto', 'log2', None],
            'max_depth': sp_randint(5, 50),
            'min_samples_leaf': sp_randint(1, 15),
            'min_samples_split': sp_randint(2, 30),
            'criterion': ['gini', 'entropy'],
            'random_state': [100]
        }

        # fitting the grid search or random search
        self._fit_grid_random_search(self.X, self.y, classifier, rf_params)


# %%
from sklearn.linear_model import LogisticRegression

print(LogisticRegression().__class__.__name__)

# %%
