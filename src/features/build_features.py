import pandas
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
from more_itertools import locate
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Filtermethods:

    @staticmethod
    def fishers_score(data, labels):
        """
        Computes the fishers for every feature between data points and label.
        :param data: Independent variables
        :param labels: Dependent Variable
        :return: Fishers score for every Feature
        """
        data_length = len(data)
        list_of_classes = []
        for label in labels:
            if label not in list_of_classes:
                list_of_classes.append(label)

        number_of_classes = len(list_of_classes)
        print('Data contains: ', number_of_classes, ' classes.')
        numerator = 0
        denominator = 0
        columns = data.columns
        fishers_score_frame = pd.DataFrame(columns=columns)

        for column in columns:
            column_mean = np.mean(data.loc[:, column])

            for label in list_of_classes:
                indexes = list(locate(labels, lambda x: x == label))
                class_in_data = data.loc[indexes, column]
                class_mean = np.mean(class_in_data)
                class_std = np.std(class_in_data)
                class_proportion = len(indexes) / data_length
                numerator = numerator + class_proportion * \
                            (class_mean - column_mean) ** 2
                denominator = denominator + class_proportion * class_std ** 2

            if denominator != 0:
                fishers_score_frame.loc[0, column] = numerator / denominator
            else:
                fishers_score_frame.loc[0, column] = 0

        print("Fisher's score(s) has/have been computed.")
        fdf = fishers_score_frame.iloc[0].dropna().sort_values(ascending=False)
        fisher_score_df = fdf.to_frame().T
        # max_features =  fisher_score_df.to_dataframe().T
        return fisher_score_df

    @staticmethod
    def mutual_information_fs(X: pandas.Series, y: pandas.Series, feature_count: int) -> object:
        """
        # Mutual information between Independent variable and Dependent Variable.
        :return:
        :type feature_count: int
        :param X: Independent Variable
        :param y: Dependent Variable
        :param feature_count: Number of features that you want
        :return: features: return number of features, df: Mutual information Data frame.
        """
        mi = mutual_info_classif(X, y)
        mi_values = pd.Series(mi)
        print("Mutual Information values {}".format(mi))
        mi_values.index = X.columns
        mi_values.sort_values(ascending=False).plot.bar(figsize=(20, 6))
        plt.ylabel('Mutual Information')
        plt.show()
        sel_ = SelectKBest(mutual_info_classif, k=feature_count)
        sel_.fit(X, y)
        feature_mi_score = pd.Series(sel_.scores_, index=X.columns)
        df = feature_mi_score.sort_values(
            ascending=False).to_frame().head(feature_count).reset_index()
        features = X.columns[sel_.get_support()]
        print("taken features are ", set(features) == set(list(df['index'])))
        X_values = sel_.transform(X)
        return features, df
