import networkx as nx
import pandas as pd
import numbers
import matplotlib.pyplot as plt
import numpy as np

from .interfaces.interface_base_learner import BaseLearnerClassInterface
from .interfaces.interface_selector import SelectorClassInterface


from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from .extended_boosting_matrix import ExtendedBoostingMatrix
from typing import Iterable
from sklearn.tree import DecisionTreeRegressor
from .additive_model_wrapper import AdditiveModelWrapper
from matplotlib.ticker import MaxNLocator


class SingleMetalCenterPathBoost(BaseEstimator, RegressorMixin):
    def __init__(self, n_iter=100, max_path_length=10, learning_rate=0.1, BaseLearnerClass=DecisionTreeRegressor,
                 kwargs_for_base_learner=None, SelectorClass=DecisionTreeRegressor, kwargs_for_selector=None,
                 verbose=False):
        if kwargs_for_base_learner is None:
            kwargs_for_base_learner = {}
        self.n_iter = n_iter
        self.max_path_length = max_path_length
        self.learning_rate = learning_rate
        self.BaseLearnerClass = BaseLearnerClass
        self.verbose = verbose
        # very basic logic in the __init__ it is just to have a more clean code, it set kwargs with default dictionaries if no other input is given, it is possible to move the dictionaries as default parameters
        self.kwargs_for_base_learner = kwargs_for_base_learner
        self.SelectorClass = SelectorClass
        self.kwargs_for_selector = kwargs_for_selector

    def fit(self, X: list[nx.Graph], y: np.array, list_anchor_nodes_labels: list[tuple], name_of_label_attribute,
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

        self._default_kwargs_for_base_learner = {'max_depth': 3,
                                                 'random_state': 0,
                                                 'splitter': 'best',
                                                 'criterion': "squared_error"
                                                 }

        self._default_kwargs_for_selector = {'max_depth': 1,
                                             'random_state': 0,
                                             'splitter': 'best',
                                             'criterion': "squared_error"
                                             }

        self._validate_data(X=X, y=y, list_anchor_nodes_labels=list_anchor_nodes_labels,
                            name_of_label_attribute=name_of_label_attribute, eval_set=eval_set)

        self.is_fitted_ = True

        self.name_of_label_attribute_ = name_of_label_attribute

        self.paths_selected_by_epb_ = set()
        self._initialize_path_boosting(X=X,
                                       list_anchor_nodes_labels=list_anchor_nodes_labels,
                                       main_label_name=name_of_label_attribute,
                                       eval_set=eval_set)

        for n_interaction in range(self.n_iter):
            if self.verbose:
                print("iteration number: ", n_interaction + 1)

            if n_interaction == 0:
                best_path = self._find_best_path(train_ebm_dataframe=self.train_ebm_dataframe_, y=y)
            else:

                negative_gradient = AdditiveModelWrapper._neg_gradient(y=y, y_hat=np.array(
                    self.base_learner_._last_train_prediction.to_numpy()))
                best_path = self._find_best_path(train_ebm_dataframe=self.train_ebm_dataframe_,
                                                 y=pd.Series(negative_gradient))

            if self.verbose:
                print("Best path: ", best_path)

            # expand the eval set in order to contain the selected columns path
            self.expand_eval_ebm_dataframe_with_best_path(best_path=best_path, main_label_name=name_of_label_attribute,
                                                          eval_set=eval_set)

            self.base_learner_.fit_one_step(X=self.train_ebm_dataframe_, y=y, best_path=best_path,
                                            eval_set=self.eval_set_ebm_df_and_target_)

            # expand the ebm dataframe with the new columns starting from the selected path
            self._expand_ebm_dataframe(X=X, selected_path=best_path, main_label_name=name_of_label_attribute)

        self.train_mse_ = self.base_learner_.train_mse

        if eval_set is not None:
            self.eval_sets_mse_ = self.base_learner_.eval_sets_mse

        self.columns_names_ = self.train_ebm_dataframe_.columns

        return self

    def expand_eval_ebm_dataframe_with_best_path(self, best_path, main_label_name, eval_set=None):
        if eval_set is not None:
            columns_names = ExtendedBoostingMatrix.get_columns_related_to_path(best_path,
                                                                               self.train_ebm_dataframe_.columns)
            for eval_set_number, eval_set_tuple in enumerate(eval_set):
                if eval_set_tuple is None:
                    continue
                eval_set_dataset, y_eval_set = eval_set_tuple
                # find the new columns in the eval set
                missing_columns = [col for col in columns_names if
                                   col not in self.eval_set_ebm_df_and_target_[eval_set_number][0].columns]
                new_columns_for_eval_set = ExtendedBoostingMatrix.generate_new_columns_from_columns_names(
                    dataset=eval_set_dataset,
                    ebm_to_be_expanded=self.eval_set_ebm_df_and_target_[eval_set_number][0],
                    columns_names=missing_columns,
                    main_label_name=main_label_name)
                self.eval_set_ebm_df_and_target_[eval_set_number][0] = pd.concat(
                    [self.eval_set_ebm_df_and_target_[eval_set_number][0], new_columns_for_eval_set], axis=1)

    def generate_ebm_for_dataset(self, dataset: list[nx.Graph], columns_names=None):
        assert self.is_fitted_
        if columns_names is None:

            selected_path = list(self.paths_selected_by_epb_)
            columns_names = []
            for path in selected_path:
                columns_names += ExtendedBoostingMatrix.get_columns_related_to_path(path=path,
                                                                                    columns_names=self.columns_names_)
        columns_names = list(set(columns_names))
        ebm_dataframe = ExtendedBoostingMatrix.generate_new_columns_from_columns_names(dataset=dataset,
                                                                                       columns_names=columns_names,
                                                                                       main_label_name=self.name_of_label_attribute_)

        return ebm_dataframe

    def predict(self, X: list[nx.Graph] | None = None, ebm_dataframe: pd.DataFrame | None = None) -> list[
        numbers.Number]:
        assert X is not None or ebm_dataframe is not None
        assert self.is_fitted_
        if ebm_dataframe is None:
            ebm_dataframe = self.generate_ebm_for_dataset(dataset=X)
        return self.base_learner_.predict(ebm_dataframe)

    def predict_step_by_step(self, X: list[nx.Graph] | None = None, ebm_dataframe: pd.DataFrame | None = None) -> list[
        np.array]:
        assert X is not None or ebm_dataframe is not None
        assert self.is_fitted_
        if ebm_dataframe is None:
            ebm_dataframe = self.generate_ebm_for_dataset(dataset=X)
        return self.base_learner_.predict_step_by_step(ebm_dataframe)

    def evaluate(self, X: list[nx.Graph] | None = None, y=None, ebm_dataframe: pd.DataFrame | None = None):
        # it returns the evolution of the mse for each iteration
        assert y is not None
        assert X is not None or ebm_dataframe is not None
        assert self.is_fitted_
        if ebm_dataframe is None:
            ebm_dataframe = self.generate_ebm_for_dataset(dataset=X)
        return self.base_learner_.evaluate(ebm_dataframe, y)

    def _expand_ebm_dataframe(self, X: list[nx.Graph], selected_path, main_label_name: str):
        if selected_path in self.paths_selected_by_epb_:
            return
        elif len(selected_path) >= self.max_path_length:
            self.paths_selected_by_epb_.add(selected_path)
        else:
            self.paths_selected_by_epb_.add(selected_path)
            new_columns = ExtendedBoostingMatrix.new_columns_to_expand_ebm_dataframe_with_path(dataset=X,
                                                                                               selected_path=selected_path,
                                                                                               main_label_name=main_label_name,
                                                                                               df_to_be_expanded=self.train_ebm_dataframe_)
            self.train_ebm_dataframe_ = pd.concat([self.train_ebm_dataframe_, new_columns], axis=1)

    def _initialize_path_boosting(self, X, list_anchor_nodes_labels: list, main_label_name: str,
                                  eval_set: list[tuple[list[nx.Graph], Iterable]] = None):

        self.name_of_label_attribute = main_label_name

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
            for eval_tuple in eval_set:
                if eval_tuple is None:
                    self.eval_set_ebm_df_and_target_.append(None)
                    continue
                else:
                    eval_dataset, y_eval_set = eval_tuple
                    # prepare extended boosting matrix for eval dataset
                    eval_set_ebm_dataframe = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
                        dataset=eval_dataset,
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

        # check BaseLearnerClass and SelectorClass
        assert issubclass(self.BaseLearnerClass, BaseLearnerClassInterface)
        assert issubclass(self.SelectorClass, SelectorClassInterface)

        if issubclass(self.BaseLearnerClass, DecisionTreeRegressor):
            if self.kwargs_for_base_learner is None:
                self.kwargs_for_base_learner = self._default_kwargs_for_base_learner
            else:
                for key in self._default_kwargs_for_base_learner:
                    if key not in self.kwargs_for_base_learner:
                        self.kwargs_for_base_learner[key] = self._default_kwargs_for_base_learner[key]

        if issubclass(self.SelectorClass, DecisionTreeRegressor):
            if self.kwargs_for_selector is None:
                self.kwargs_for_selector = self._default_kwargs_for_selector
            else:
                for key in self._default_kwargs_for_selector:
                    if key not in self.kwargs_for_selector:
                        self.kwargs_for_selector[key] = self._default_kwargs_for_selector[key]

        list_anchor_nodes_labels = check_params.get('list_anchor_nodes_labels', None)
        if list_anchor_nodes_labels is not None:
            # Ensure each element in list_anchor_nodes_labels is a tuple
            len_list_anchor_nodes_labels = len(list_anchor_nodes_labels)
            for i in range(len_list_anchor_nodes_labels):
                if not isinstance(list_anchor_nodes_labels[i], tuple):
                    if hasattr(list_anchor_nodes_labels[i], '__iter__') and not isinstance(list_anchor_nodes_labels[i],
                                                                                           str):
                        list_anchor_nodes_labels[i] = tuple(list_anchor_nodes_labels[i])
                    else:
                        list_anchor_nodes_labels[i] = tuple([list_anchor_nodes_labels[i]])

    def plot_training_and_eval_errors(self):
        """
        Plots the training and evaluation set errors over iterations.
        """
        # skip_the_first n iterations
        n = int(2 / self.learning_rate)
        train_mse = self.train_mse_[n:]

        plt.figure(figsize=(12, 6))

        # Plot training errors
        plt.plot(range(n, len(train_mse) + n), train_mse, label='Training Error', marker='o')

        # Plot evaluation set errors if available
        if hasattr(self, 'eval_sets_mse_'):
            eval_sets_mse = self.eval_sets_mse_[n:]
            num_iterations = len(eval_sets_mse[0])
            num_eval_sets = len(eval_sets_mse)
            for eval_set_index in range(num_eval_sets):
                if eval_sets_mse[eval_set_index][0] is not None:
                    plt.plot(range(n, num_iterations + n), eval_sets_mse[eval_set_index],
                             label=f'Evaluation Set {eval_set_index + 1} Error', marker='.')

        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('Training and Evaluation Set Errors Over Iterations')
        plt.legend()
        plt.grid(True)

        # Ensure x-axis only shows integers
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()
