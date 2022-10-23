from src.features.build_features import FilterMethods
from src.models.model_fitting import ParameterTuning


class ModelFittingPipeLine(ParameterTuning):
    """
    Model fitting Pipe line for all classifiers.
    """

    def __init__(self, X, y,
                 feature_selection_method_name='fisher_score',
                 metric_type='accuracy',
                 file_location="",
                 search_type='grid_search'):
        super().__init__(X, y, metric_type, file_location, search_type)
        self.feature_selection_method_name = feature_selection_method_name

