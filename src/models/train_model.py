from src.features.build_features import FilterMethods
from src.models.model_fitting import ModelFittingPipeLine


class ModelFitting(ModelFittingPipeLine, FilterMethods):
    """
    Model fitting Pipe line for all classifiers.
    """

    def __init__(self, X, y,
                 feature_selection_method_name='fisher_score',
                 number_of_features=5,
                 metric_type='accuracy',
                 file_location="",
                 search_type='grid_search'):
        ModelFittingPipeLine.__init__(self, X, y, metric_type, file_location, search_type)
        self.feature_selection_method_name = feature_selection_method_name,
        self.number_of_features = number_of_features

    def fitting(self):
        """
        it will be fitting the models and return dictionary of models.
        """
        # Feature Selection
        X = self.X[0]
        y = self.y[0]
        feature_selection_method_name = self.feature_selection_method_name[0]
        number_of_features = self.number_of_features[0]
        print("--" * 50)
        print("1.Feature Selection ")
        features = FilterMethods().feature_selection_type(X, y, feature_selection_method_name, number_of_features)
        X_data = X[features]


# %%

