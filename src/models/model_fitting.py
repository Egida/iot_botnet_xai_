from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics


class tuning_algorithmms:
    """
    Tuning Algorithms
    """
    def __int__(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """
        self.X = X,
        self.y = y

    def _fit_grid_random_search(self, X, y, model, parameters):

        """ Training the model using Grid search or Random search hyperparameter tuning methods.

        :param X: Independent Variable
        :param y: Dependent Variable
        :param model: Scikit learn classifier
        :param parameters: various combinations for parameters for classifier.
        :return:
        """
        with open('reports/parameter_tuning', 'a') as results:
            results.write()

        print("hewllo world ")
        return 0
