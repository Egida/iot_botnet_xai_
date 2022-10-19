import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from numpy import mean
import pandas as pandas
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import randint as sp_randint
from random import randrange as sp_randrange
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class parameter_tuning:
    """
    Tuning Algorithms
    """

    def __int__(self, X, y, metric_type, file_location, search_type='grid_search'):
        """
        Tuning the algorithms

        :param file_location: resultant file location.
        :param metric_type: 1. accuracy, f1-score, recall, precision
        :param cv: Cross validation score
        :param search_type: Hyper Parameter Tuning type 1. Grid Search 2. Random Search type
        :return:
        """
        self.X = X,
        self.y = y,
        self.metric_type = metric_type,
        self.search_type = search_type,
        self.file_location = file_location

    def _fit_grid_random_search(self, X, y, ml_classifier, parameters):

        """ Training the model using Grid search or Random search hyperparameter tuning methods.

        :param X: Independent Variable
        :param y: Dependent Variable
        :param ml_classifier: Scikit learn classifier
        :param parameters: various combinations for parameters for classifier.
        :return:
        """

        print("Tuning Type:{0}\n".format(self.search_type))
        # Classifier name
        mlclassifier_name = str(type(ml_classifier)).split(".")[-1][:-2]
        print("Classifier is {0}".format(mlclassifier_name))
        # check the Parameter type,
        cv = KFold(n_splits=5, random_state=100, shuffle=True)

        if self.search_type == 'grid_search':
            # Grid Search parameter type
            tuned_model = GridSearchCV(ml_classifier,
                                       param_grid=parameters,
                                       scoring=self.metric_type,
                                       verbose=10,
                                       refit=False)
        elif self.search_type == 'random_search':
            # Random Search Parameter Tuning
            tuned_model = RandomizedSearchCV(estimator=ml_classifier,
                                             param_distributions=parameters,
                                             scoring=self.metric_type,
                                             cv=cv,
                                             verbose=10,
                                             refit=False)

        else:
            print("===========================================")
            print(f'{search_type} is wrong key word.'
                  f'Key word should be either 1.grid_search or 2.random_search')

        # Finally fit the Classifier
        start_time = self.timer(0)  # starting time for model training
        tuned_model = tuned_model.fit(self.X[0], self.y[0])  # fitting the model
        finish_time = self.timer(start_time)  # Finishing for model training

        # Save the model
        joblib.dump(tuned_model, f'reports/result_logs/{self.file_location}/{mlclassifier_name}.pkl')

        # adding output results to a file.
        with open('reports/parameter_tuning.txt', 'a') as res_logs:
            res_logs.write('==' * 40)
            res_logs.write("\n")
            res_logs.write("1.Classifier:{0}\n".format(mlclassifier_name))
            res_logs.write("2.Best Parameters:{0}\n".format(str(tuned_model.best_params_)))
            res_logs.write("3.Duration:{0}\n".format(str(finish_time)))
            res_logs.write('\nAccuracy: %.5f (%.5f)' % (tuned_model.best_score_ * 100, mean(tuned_model.cv_results_['std_test_score']) * 100))
            res_logs.write('\n')
            res_logs.write('==' * 40)
            res_logs.write('\n')
        return tuned_model.best_score_

    # Time to  count the model for training.
    @staticmethod
    def timer(start_time=None):
        """

        :param start_time: 0
        :return: Completion time
        """
        time_list = []
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            # time_list.append(thour)
            # print("\n Time taken: %i hours %i minutes and %s seconds" % (thour, tmin, round(tsec,2)))
        return str("Time consumption: %i hours %i minutes and %s seconds" % (thour, tmin, round(tsec, 2)))

    def rf_classification(self):
        """
        Random Forest Classifier
        """
        # Initiate the classifier
        classifier = RandomForestClassifier(n_jobs=-1)
        # parameters
        rf_params = {
            'max_features': ['sqrt', 'auto', 'log2', None],
            'max_depth': sp_randint(5, 50),
            'min_samples_leaf': sp_randint(1, 15),
            'min_samples_split': sp_randint(2, 30),
            'criterion': ['gini', 'entropy'],
            'random_state': [100]
        }
        print("Tuning Type:{0}\n".format(self.search_type))
        print("Classifier name:{0}\n".format(classifier.__class__.__name__))
        for key, value in rf_params.items():
            print("{0}:{1}".format(key,value))
        # parameters for grid search
        # fitting the grid search or random search
        self._fit_grid_random_search(self.X[0], self.y[0], classifier, rf_params)

    def dt_classification(self):
        """
        Decision Tree Classifier
        """
        # Initiate the classifier
        classifier = DecisionTreeClassifier(n_jobs=-1)
        # parameters
        dt_params = {
            'max_features': ['sqrt', 'auto', 'log2', None],
            'max_depth': sp_randint(5, 50),
            'min_samples_leaf': sp_randint(1, 15),
            'min_samples_split': sp_randint(2, 30),
            'criterion': ['gini', 'entropy'],
            'random_state': [100]
        }
        print("Tuning Type:{0}\n".format(self.search_type))
        print("Classifier name:{0}\n".format(classifier.__class__.__name__))
        for key, value in dt_params.items():
            print("{0}:{1}".format(key, value))
        # parameters for grid search
        # fitting the grid search or random search
        self._fit_grid_random_search(self.X[0], self.y[0], classifier, dt_params)

    def knn_classification(self):
        """
K-nearest neighbor classification
        """
        # Initiate the classifier
        classifier = KNeighborsClassifier(n_jobs=-1)
        # parameters
        knn_params = {
            'max_features': ['sqrt', 'auto', 'log2', None],
            'max_depth': sp_randint(5, 50),
            'min_samples_leaf': sp_randint(1, 15),
            'min_samples_split': sp_randint(2, 30),
            'criterion': ['gini', 'entropy'],
            'random_state': [100]
        }
        # print("Tuning Type:{0}\n".format(self.search_type))
        # print("Classifier name:{0}\n".format(classifier.__class__.__name__))
        for key, value in knn_params.items():
            print("{0}:{1}".format(key, value))
        # parameters for grid search
        # fitting the grid search or random search
        self._fit_grid_random_search(self.X[0], self.y[0], classifier, knn_params)






# %%
from sklearn.linear_model import LogisticRegression

print(LogisticRegression().__class__.__name__)

# %%
