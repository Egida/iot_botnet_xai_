from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from numpy import mean
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import uniform as sp_uniform
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier


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

    def _fit_grid_random_search(self, ml_classifier, parameters):

        """ Training the model using Grid search or Random search hyperparameter tuning methods.
        :param X: Independent Variable
        :param y: Dependent Variable
        :param ml_classifier: Scikit learn classifier
        :param parameters: various combinations for parameters for classifier.
        :return:
        """

        cv_results_df = pd.DataFrame()
        print("Tuning Type:{0}\n".format(self.search_type))
        # Classifier name
        mlclassifier_name = str(type(ml_classifier)).split(".")[-1][:-2]
        print("Classifier is {0}".format(mlclassifier_name))
        X = self.X[0]
        y = self.y[0]
        # data shape
        print("X variable: {0}\ny Variable:{1}".format(X.shape, y.shape))
        # check the Parameter type,
        cv = KFold(n_splits=5, random_state=100, shuffle=True)

        search_type = self.search_type[0]
        # Grid search tuning.
        if search_type == 'grid_search':
            # Grid Search parameter type
            tuned_model = GridSearchCV(ml_classifier,
                                       param_grid=parameters,
                                       scoring=self.metric_type,
                                       verbose=10,
                                       refit=False)
            start_time = self.timer(0)
            tuned_model.fit(X, y)
            finishing_time = self.timer(start_time)
            print("Best parameters:{0}".format(tuned_model.best_params_))
            # saving the logs of model into a text file
            df = self.res_logs_text_file(mlclassifier_name,
                                         tuned_model,
                                         finishing_time,
                                         self.file_location)

            return cv_results_df.append(df)
        # random search
        elif search_type == 'random_search':
            # Random Search Parameter Tuning
            tuned_model = RandomizedSearchCV(estimator=ml_classifier, param_distributions=parameters,
                                             scoring=self.metric_type, cv=cv,
                                             verbose=10, refit=False)
            # Tuning the model
            start_time = self.timer(0)
            tuned_model.fit(X, y)
            finishing_time = self.timer(start_time)
            print("Best parameters:{0}".format(tuned_model.best_params_))
            # saving the logs of model into a text file
            df = self.res_logs_text_file(mlclassifier_name,
                                         tuned_model,
                                         finishing_time,
                                         self.file_location)
            return cv_results_df.append(df)
        else:
            print("===========================================")
            print(f'{search_type} is wrong key word.'
                  f'Key word should be either 1.grid_search or 2.random_search')

        # save the model
        return cv_results_df

    def res_logs_text_file(self, mlclassifier_name, tuned_model, finish_time, file_location):
        """
        saving the result into a text files
        :param file_location: save resultant file location name. it must be with dataset name
        :param mlclassifier_name: classifier name
        :param tuned_model:  trained model
        :param finish_time: model finishing time
        :return: dataframe
        """
        with open(f'reports/{file_location}/parameter_tuning.txt', 'a') as res_logs:
            res_logs.write('==' * 40)
            res_logs.write("\n")
            res_logs.write("1.Classifier:{0}\n".format(mlclassifier_name))
            res_logs.write("2.Best Parameters:{0}\n".format(str(tuned_model.best_params_)))
            res_logs.write("3.Duration:{0}\n".format(str(finish_time)))
            res_logs.write('\nAccuracy: %.5f (%.5f)' % (
                tuned_model.best_score_ * 100, mean(tuned_model.cv_results_['std_test_score']) * 100))
            res_logs.write('\n')
            res_logs.write('==' * 40)
            res_logs.write('\n')

        # cv results
        cv_results_df = pd.DataFrame(tuned_model.cv_results_)

        # save the model
        file_name = f'models/{self.file_location}/{mlclassifier_name}.pkl'
        joblib.dump(tuned_model, file_name)

        return cv_results_df

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
            'max_depth': list(range(5, 51)),
            'min_samples_leaf': list(range(1, 16)),
            'min_samples_split': list(range(2, 31)),
            'criterion': ['gini', 'entropy'],
            'random_state': [100]
        }

        print("Tuning Type:{0}\n".format(self.search_type))
        print("Classifier name:{0}\n".format(classifier.__class__.__name__))
        for key, value in rf_params.items():
            print("{0}:{1}".format(key, value))
        # parameters for grid search
        # fitting the grid search or random search
        cv_results = self._fit_grid_random_search(classifier, rf_params)
        return cv_results

    def dt_classification(self):
        """
        Decision Tree Classifier
        """
        # Initiate the classifier
        classifier = DecisionTreeClassifier()
        # parameters
        dt_params = {
            'max_features': ['sqrt', 'auto', 'log2', None],
            'max_depth': list(range(5, 51)),
            'min_samples_leaf': list(range(1, 16)),
            'min_samples_split': list(range(2, 31)),
            'criterion': ['gini', 'entropy'],
            'random_state': [100]
        }

        # print("Tuning Type:{0}\n".format(self.search_type))
        # print("Classifier name:{0}\n".format(classifier.__class__.__name__))
        # for key, value in dt_params.items():
        #     print("{0}:{1}".format(key, value))
        # parameters for grid search
        # fitting the grid search or random search
        cv_results = self._fit_grid_random_search(classifier, dt_params)
        return cv_results

    def knn_classification(self):
        """
K-nearest neighbor classification
        """
        # Initiate the classifier
        classifier = KNeighborsClassifier(n_jobs=-1)
        # parameters
        k_range = list(range(1, 31))
        knn_params = {
            'n_neighbors': list(range(1, 21, 1)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        # print("Tuning Type:{0}\n".format(self.search_type))
        # print("Classifier name:{0}\n".format(classifier.__class__.__name__))
        for key, value in knn_params.items():
            print("{0}:{1}".format(key, value))
        # parameters for grid search
        # fitting the grid search or random search
        cv_results = self._fit_grid_random_search(classifier, knn_params)
        return cv_results

    def xgboost_classification(self):
        """
xgboost
        """
        xgb_params = rf_params = {
            'num_leaves': sp_randint(6, 50),
            'min_child_samples': sp_randint(100, 500),
            'learning_rate': list(np.arange(0, 1.1, 0.4)),
            'max_depth': list(range(5, 51, 5)),
            'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
            'subsample': sp_uniform(loc=0.2, scale=0.8),
            'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
            'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
            'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
        }
        xgb_classifier = xgb.XGBClassifier(objective='binary:logistic',
                                           use_label_encoder=False,
                                           random_state=100)

        cv_results = self._fit_grid_random_search(xgb_classifier, xgb_params)
        return cv_results

    def lgboost_classification(self):
        """
        Light gradient boosting
        """
        # parameters combinations
        lgb_params = {
            'num_leaves': sp_randint(6, 50),
            'learning_rate': list(np.arange(0, 1.1, 0.4)),
            'min_child_samples': sp_randint(100, 500),
            'max_depth': list(range(5, 51, 5)),
            'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
            'subsample': sp_uniform(loc=0.2, scale=0.8),
            'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
            'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
            'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
        }

        lgbm_classifier = lgb.LGBMClassifier(random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)

        cv_results = self._fit_grid_random_search(lgbm_classifier, lgb_params)
        return cv_results

    def et_classification(self):
        """
        Extra tree Classification
        """
        xt_clf = ExtraTreesClassifier(verbose=10,
                                      random_state=123,
                                      n_jobs=-1)

        xt_params = {
            'n_estimators': [int(x) for x in range(200, 2000, 200)],
            'max_features': ['sqrt', 'auto', 'log2', None],
            'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
            'min_samples_leaf': sp_randint(1, 15),
            'min_samples_split': sp_randint(2, 30),
            'bootstrap': [True, False]
        }

        cv_results = self._fit_grid_random_search(xt_clf, xt_params)
        return cv_results

    def grdient_boosting_classification(self):
        """
        Gradient Boosting classifier
        """
        lgb_params = {
            'num_leaves': sp_randint(6, 50),
            'min_child_samples': sp_randint(100, 500),
            'max_depth': list(range(5, 51, 5)),
            'learning_rate': list(np.arange(0, 1.1, 0.4)),
            'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
            'subsample': sp_uniform(loc=0.2, scale=0.8),
            'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
            'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
            'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
        }

        gbc_clf = GradientBoostingClassifier(min_samples_split=500, min_samples_leaf=50, max_depth=8,
                                             max_features='sqrt', subsample=0.8, random_state=10)
        cv_results = self._fit_grid_random_search(lgb_params, gbc_clf)
        return cv_results

    def fitting_models(self):
        """
        fitting all the models
        """
        model_fitting_dict = {'dt': self.dt_classification(),
                              'rf': self.rf_classification(),
                              'ext': self.et_classification(),
                              'gbc': self.grdient_boosting_classification(),
                              'xgb': self.xgboost_classification(),
                              'lgb': self.lgboost_classification(),
                              'knn': self.knn_classification()
                              }
        return model_fitting_dict
