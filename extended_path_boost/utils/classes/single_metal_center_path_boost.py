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
                 kwargs_for_base_learner=None, Selector=DecisionTreeRegressor, kwargs_for_selector=None):
        if kwargs_for_base_learner is None:
            kwargs_for_base_learner = {}
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

    def fit(self, X: list[nx.Graph], y: np.array, list_anchor_nodes_labels: list, name_of_label_attribute,
            eval_set: list[tuple[list[nx.Graph], Iterable]] = None):
        """
        Fits the boosting algorithm to the input data, performing a series of iterative steps
        to identify the best paths and train the underlying model. It initializes the path
        boosting process, computes gradients, and updates the model iteratively using the
        defined number of iterations. The method also stores and updates intermediate
        results like selected paths and mean squared errors for training and evaluation sets.

        Parameters:
            X: list[nx.Graph]
                A list of networkx graph objects used as input data for training.
            y: np.array
                A numpy array containing target values for supervised learning.
            list_anchor_nodes_labels: list
                A list that contains the values of the attribute "name_of_label_attribute" for each Anchor node
            name_of_label_attribute: Any
                Attribute name associated with the node labels.
            eval_set: list[tuple[list[nx.Graph], Iterable]], optional
                A list of tuples comprising evaluation datasets where each tuple contains
                a list of graphs and their corresponding target values.

        Returns:
            self: object
                The fitted object instance.

        Raises:
            None, the function assumes all preprocessing and validation steps have been
            handled externally.
        """

        self._validate_data(X=X, y=y, list_anchor_nodes_labels=list_anchor_nodes_labels,
                            name_of_label_attribute=name_of_label_attribute, eval_set=eval_set)

        self.is_fitted_ = True

        self.paths_selected_by_epb_ = set()
        self._initialize_path_boosting(X=X,
                                       list_anchor_nodes_labels=list_anchor_nodes_labels,
                                       main_label_name=name_of_label_attribute,
                                       eval_set=eval_set)

        for n_interaction in range(self.n_iter):
            print("iteration number: ", n_interaction + 1)

            if n_interaction == 0:
                best_path = self._find_best_path(train_ebm_dataframe=self.train_ebm_dataframe_, y=y)
            else:

                negative_gradient = AdditiveModelWrapper._neg_gradient(y=y, y_hat=np.array(
                    self.base_learner_._last_train_prediction.to_numpy()))
                best_path = self._find_best_path(train_ebm_dataframe=self.train_ebm_dataframe_,
                                                 y=pd.Series(negative_gradient))

            self.base_learner_.fit_one_step(X=self.train_ebm_dataframe_, y=y, best_path=best_path,
                                            eval_set=self.eval_set_ebm_df_and_target_)

            # expand the ebm dataframe with the new columns starting from the selected path
            self._expand_ebm_dataframe(X=X, selected_path=best_path, main_label_name=name_of_label_attribute,
                                       eval_set=eval_set)

        self.train_mse_ = self.base_learner_.train_mse
        if eval_set is not None:
            self.eval_sets_mse_ = self.base_learner_.eval_sets_mse

        return self

    def predict(self, dataset):
        return

    def score(self, dataset):
        return

    def _expand_ebm_dataframe(self, X: list[nx.Graph], selected_path, main_label_name: str, eval_set=None):
        if selected_path in self.paths_selected_by_epb_ or len(selected_path) >= self.max_path_length:
            return
        else:
            self.paths_selected_by_epb_.add(selected_path)
            new_columns = ExtendedBoostingMatrix.new_columns_to_expand_ebm_dataframe_with_path(dataset=X,
                                                                                               selected_path=selected_path,
                                                                                               main_label_name=main_label_name,
                                                                                               df_to_be_expanded=self.train_ebm_dataframe_)
            self.train_ebm_dataframe_ = pd.concat([self.train_ebm_dataframe_, new_columns], axis=1)

            # add them also to the eval set ebm
            if eval_set is not None:

                for eval_set_number, eval_set_dataset_target in enumerate(eval_set):
                    eval_set_dataset, y_eval_set = eval_set_dataset_target
                    # find the new columns in the eval set

                    new_columns_for_eval_set = ExtendedBoostingMatrix.generate_new_columns_from_columns_names(
                        dataset=eval_set_dataset,
                        ebm_to_be_expanded=self.eval_set_ebm_df_and_target_[eval_set_number][0],
                        columns_names=new_columns.columns,
                        main_label_name=main_label_name)
                    self.eval_set_ebm_df_and_target_[eval_set_number][0] = pd.concat(
                        [self.eval_set_ebm_df_and_target_[eval_set_number][0], new_columns_for_eval_set], axis=1)

    def _initialize_path_boosting(self, X, list_anchor_nodes_labels: list, main_label_name: str,
                                  eval_set: list[tuple[list[nx.Graph], Iterable]] = None):

        # greate extended boosting matrix for train dataset
        self.train_ebm_dataframe_ = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
            dataset=X,
            list_anchor_nodes_labels=list_anchor_nodes_labels,
            id_label_name=main_label_name)
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
                    id_label_name=main_label_name)
                self.eval_set_ebm_df_and_target_.append([eval_set_ebm_dataframe, y_eval_set])

        # initialize base learner wrapper
        self.base_learner_: AdditiveModelWrapper = AdditiveModelWrapper(BaseModelClass=self.BaseLearnerClass,
                                                                        base_model_class_kwargs=self.kwargs_for_base_learner,
                                                                        learning_rate=self.learning_rate, )

    def _find_best_path(self, train_ebm_dataframe: pd.DataFrame, y) -> tuple[int]:
        base_feature_selector = self.SelectorClass(**self.kwargs_for_selector)
        frequency_boosting_matrix = ExtendedBoostingMatrix.get_frequency_boosting_matrix(train_ebm_dataframe)

        base_feature_selector = base_feature_selector.fit(X=frequency_boosting_matrix, y=y)
        best_feature_index = np.array(base_feature_selector.feature_importances_).argmax()
        best_feature = frequency_boosting_matrix.columns[best_feature_index]
        best_path = ExtendedBoostingMatrix.get_path_from_column_name(best_feature)
        return best_path

    def _validate_data(
            self,
            X: list[nx.Graph] = "no_validation",
            y="no_validation",
            **check_params,
    ):
        if isinstance(X, str) and X == "no_validation":
            raise ValueError("X is not provided")
        if isinstance(y, str) and y == "no_validation":
            raise ValueError("y is not provided")
