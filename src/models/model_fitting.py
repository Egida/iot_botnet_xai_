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
from src.features.build_features import FilterMethods


class ModelFittingPipeLine(FilterMethods):
    """
    Tuning Algorithms
    """

    def __int__(self, X, y, metric_type, feature_selection_method_name='fisher_score', number_of_features=5,
                search_type='grid_search', file_location=""):
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
        self.feature_selection_method_name = feature_selection_method_name,
        self.number_of_features = number_of_features,
        self.metric_type = metric_type,
        self.search_type = search_type,
        self.file_location = file_location

    def _fit_grid_random_search(self, ml_classifier, parameters):

        """ Training the model using Grid search or Random search hyperparameter tuning methods.
        :param X: Independent Variable
        :param y: Dependent Variable
        :param ml_classifier: Scikit learn classifier
        :param parameters: various combinations for parameters for classifier.
        :return: (dict) resultant dict.
        """

        cv_results_df = pd.DataFrame()
        print("==" * 50)
        # classifier Name
        mlclassifier_name = str(type(ml_classifier)).split(".")[-1][:-2]
        print("\nClassifier is {0}\n".format(mlclassifier_name))
        # changing tuple object to variables
        X = self.X[0]
        y = self.y[0]
        feature_selection_method_name = self.feature_selection_method_name[0]
        number_of_features = self.number_of_features[0]
        # data shape
        print("X variable: {0}\ny Variable:{1}\n".format(X.shape, y.shape))

        # feature selection
        features = self.feature_selection_type(X, y, feature_selection_method_name, number_of_features)
        X = X[features]
        print("--" * 40)
        print("after Feature Selection\n")
        print("data shape:{0}\n".format(X.shape))
        # cross validation
        cv = KFold(n_splits=10, random_state=100, shuffle=True)
        search_type = self.search_type[0]
        metric_type = self.metric_type[0]
        file_location = self.file_location[0]
        print("Metric:{0}".format(metric_type))
        print("Tuning Type:{0}\n".format(search_type))
        print("Parameters are:")
        for key, value in parameters.items():
            print("{0}:{1}".format(key, value))

        # Grid search tuning.
        if search_type == 'grid_search':
            # Grid Search parameter type
            tuned_model = GridSearchCV(estimator=ml_classifier,
                                       param_grid=parameters,
                                       scoring=metric_type,
                                       verbose=10, refit='AUC',
                                       return_train_score=True, n_jobs=-1)
            start_time = self.timer(0)
            tuned_model.fit(X, y)
            finishing_time = self.timer(start_time)

            file_name = f'{self.file_location}/{mlclassifier_name}.pkl'
            print("file Location with name {0} ".format(file_name))
            joblib.dump(tuned_model.best_estimator_, file_name)

            print("Best parameters:{0}".format(tuned_model.best_params_))
            print("Best Estimator:{0}".format(tuned_model.best_estimator_))
            # saving the logs of model into a text file
            df = self.res_logs_text_file(mlclassifier_name,
                                         tuned_model,
                                         finishing_time,
                                         file_location)
            model_res_dict = {mlclassifier_name: cv_results_df.append(df)}
            return model_res_dict
        # random search
        elif search_type == 'random_search':
            # Random Search Parameter Tuning
            tuned_model = RandomizedSearchCV(estimator=ml_classifier, param_distributions=parameters,
                                             scoring=metric_type, cv=cv,
                                             verbose=10, refit='AUC', return_train_score=True, n_jobs=-1)
            # Tuning the model
            start_time = self.timer(0)
            tuned_model.fit(X, y)
            finishing_time = self.timer(start_time)
            print("Best Parameters:{0}".format(tuned_model.best_params_))
            print("Best Estimator:{0}".format(tuned_model.best_estimator_))
            file_name = f'{self.file_location}/{mlclassifier_name}.pkl'
            joblib.dump(tuned_model.best_estimator_, file_name)
            # saving the logs of model into a text file
            df = self.res_logs_text_file(mlclassifier_name,
                                         tuned_model,
                                         finishing_time,
                                         file_location)
            return cv_results_df.append(df)
        else:
            print("===========================================")
            print(f'{search_type} is wrong key word.'
                  f'Key word should be either 1.grid_search or 2.random_search')

        # save the model
        return cv_results_df

    @staticmethod
    def res_logs_text_file(mlclassifier_name, tuned_model, finish_time, file_location):
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
            res_logs.write("4.Best Estimator{0}\n".format(str(tuned_model.best_estimator_)))
            res_logs.write('\nAccuracy: %.5f ' % (tuned_model.best_score_ * 100))
            res_logs.write('\n')
            res_logs.write('==' * 40)
            res_logs.write('\n')

        # cv results
        cv_results_df = pd.DataFrame(tuned_model.cv_results_)

        # save the model
        return cv_results_df

    # Time to  count the model for training.
    @staticmethod
    def timer(start_time=None):
        """
        :param start_time: 0
        :return: Completion time string
        """
        resultant_string = ""
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            resultant_string += str("Time consumption: %i hours %i minutes and %s "
                                    "seconds" % (thour, tmin, round(tsec, 2)))
            # time_list.append(thour)
            # print("\n Time taken: %i hours %i minutes and %s seconds" % (thour, tmin, round(tsec,2)))
        return resultant_string

    def rf_classification(self):
        """
        Random Forest Classifier
        """
        # Initiate the classifier
        classifier = RandomForestClassifier(n_jobs=-1)
        # Classifier name

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
        # Initiate Classifier
        lgbm_classifier = lgb.LGBMClassifier(random_state=314,
                                             silent=True, metric='None',
                                             n_jobs=4, n_estimators=5000)

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

        cv_results = self._fit_grid_random_search(lgbm_classifier, lgb_params)
        return cv_results

    def et_classification(self):
        """
        Extra tree Classification
        """
        # Initiate classifier
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
        gbc_params = {
            'n_estimators': [int(x) for x in range(200, 2000, 200)],
            'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
            'learning_rate': [0.1, 0.001, 0.01]
        }

        gbc_clf = GradientBoostingClassifier(min_samples_split=500, min_samples_leaf=50, max_depth=8,
                                             max_features='sqrt', subsample=0.8, random_state=10)
        cv_results = self._fit_grid_random_search(gbc_clf, gbc_params)
        return cv_results

    def fitting_models(self):
        """
        fitting all the models
        """
        file_location = self.file_location[0]
        # initiate the dict of models.
        file_name = f'{file_location}/all_models_cv_results.pkl'
        model_fitting_dict = {}
        model_fitting_dict.update(self.dt_classification())
        model_fitting_dict.update(self.rf_classification())
        model_fitting_dict.update(self.et_classification())
        model_fitting_dict.update(self.xgboost_classification())
        model_fitting_dict.update(self.lgboost_classification())
        model_fitting_dict.update(self.knn_classification())
        model_fitting_dict.update(self.grdient_boosting_classification())
        joblib.dump(model_fitting_dict,file_name)
        return model_fitting_dict

    def fitting_with_feature_selection(self, X, y, feature_Selection_method_name, number_of_features):
        """
        Fitting with feature selection
        """
        # read the features

        features = self.feature_selection_type(X, y,
                                               feature_Selection_method_name,
                                               number_of_features)
        return features
