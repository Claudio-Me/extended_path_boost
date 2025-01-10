import networkx as nx
import pandas as pd
from sklearn.base import BaseEstimator
import numpy as np
from .extended_boosting_matrix import ExtendedBoostingMatrix
from typing import Iterable
from sklearn.tree import DecisionTreeRegressor
from .additive_model_wrapper import AdditiveModelWrapper


class SingleMetalCenterPathBoost(BaseEstimator):
    def __init__(self, n_iter=100, max_path_length=10, learning_rate=0.1, BaseLearner=DecisionTreeRegressor,
                 kwargs_for_base_learner=None, Selector=DecisionTreeRegressor, kwargs_for_selector=None,
                 base_learner_parameters=None):
        self.n_iter = n_iter
        self.max_path_length = max_path_length
        self.learning_rate = learning_rate
        self.BaseLearnerClass = BaseLearner
        # very basic logic in the __init__ it is just to have a more clean code, it set kwargs with default dictionaries if no other input is given, it is possible to move the dictionaries as default parameters
        self.kwargs_for_base_learner = kwargs_for_base_learner or {'max_depth': 3, 'random_state': 0,
                                                                   'splitter': 'best', 'criterion': "squared_error"}
        self.SelectorClass = Selector
        self.kwargs_for_selector = kwargs_for_selector or {'max_depth': 1, 'random_state': 0, 'splitter': 'best',
                                                           'criterion': "squared_error"}
        self.base_learner_parameters = base_learner_parameters

    def fit(self, X: list[nx.Graph], y: np.array, list_anchor_nodes_labels, name_of_label_attribute,
            eval_set: list[tuple[list[nx.Graph], Iterable]] = None):
        self.is_fitted_ = True
        self._initialize_path_boosting(X=X,
                                       list_anchor_nodes_labels=list_anchor_nodes_labels,
                                       id_label_name=name_of_label_attribute,
                                       eval_set=eval_set)

        for n_interaction in range(self.n_iter):
            print("iteration number: ", n_interaction + 1)
            if n_interaction == 0:
                best_path = self._find_best_path(self.train_ebm_dataframe_, y)

                self.base_learner_.fit_one_step(X=self.train_ebm_dataframe_, y=y, best_path=best_path,
                                                eval_set=self.eval_set_ebm_df_and_target_)

        # `fit` should always return `self`
        return self

    def predict(self, dataset):
        return

    def score(self, dataset):
        return

    def _initialize_path_boosting(self, X, list_anchor_nodes_labels: list, id_label_name: str,
                                  eval_set: list[tuple[list[nx.Graph], Iterable]] = None):

        # greate extended boosting matrix for train dataset
        self.train_ebm_dataframe_ = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
            dataset=X,
            list_anchor_nodes_labels=list_anchor_nodes_labels,
            id_label_name=id_label_name)
        self.eval_set_ebm_df_and_target_ = []

        # generate extended boosting matrix for eval dataset
        if eval_set is None:
            pass
        else:
            for dataset, y_eval_set in eval_set:
                # prepare extended boosting matrix for eval dataset
                eval_set_ebm_dataframe = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
                    dataset=dataset,
                    list_anchor_nodes_labels=list_anchor_nodes_labels,
                    id_label_name=id_label_name)
                self.eval_set_ebm_df_and_target_.append([eval_set_ebm_dataframe, y_eval_set])

        # initialize base learner wrapper
        self.base_learner_: AdditiveModelWrapper = AdditiveModelWrapper(BaseModelClass=self.BaseLearnerClass,
                                                                        base_model_class_kwargs=self.base_learner_parameters,
                                                                        learning_rate=self.learning_rate, )

    def _find_best_path(self, train_ebm_dataframe: pd.DataFrame, y) -> tuple[int]:
        base_feature_selector = self.SelectorClass(**self.kwargs_for_selector)
        frequency_boosting_matrix = ExtendedBoostingMatrix.get_frequency_boosting_matrix(train_ebm_dataframe)

        base_feature_selector = base_feature_selector.fit(X=frequency_boosting_matrix, y=y)
        best_feature_index = np.array(base_feature_selector.feature_importances_).argmax()
        best_feature = frequency_boosting_matrix.columns[best_feature_index]
        best_path = ExtendedBoostingMatrix.get_path_from_column_name(best_feature)
        return best_path
