"""
Extended Path Boost - A gradient boosting algorithm for graph-structured data.

This module provides the PathBoost class, an ensemble learning method that builds
interpretable models for graph data by discovering path-based features.
"""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause


import os
import logging

# done to limit the number of spawned threads during parallelization

max_n_threads = 2
os.environ["MKL_NUM_THREADS"] = str(max_n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(max_n_threads)
os.environ["OMP_NUM_THREADS"] = str(max_n_threads)

import numbers
import numpy as np
import warnings
import itertools
import multiprocessing as mp
import networkx as nx
import matplotlib.pyplot as plt

from .utils.classes.sequential_path_boost import SequentialPathBoost
from .utils import cyclic_path_boost_utils as wbu
from .utils.classes.interfaces.interface_base_learner import BaseLearnerClassInterface
from .utils.variable_importance_according_to_path_boost import VariableImportance_ForSequentialPathBoost
from .utils.classes.interfaces.interface_selector import SelectorClassInterface
from .utils.validate_data import util_validate_data
from .utils.plots_functions import plot_training_and_eval_errors, plot_variable_importance_utils
from typing import Iterable, List, Tuple, Optional, Union, Dict, Any, Type
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, _fit_context
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import RegressorMixin

# Type aliases for better readability
GraphList = List[nx.Graph]
AnchorLabel = Union[int, str, Tuple]
AnchorLabelList = List[AnchorLabel]
EvalSet = List[Tuple[GraphList, Iterable]]

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Set up logger for the module
logger = logging.getLogger('path_boost')


class PathBoost(BaseEstimator, RegressorMixin):
    """
    PathBoost is an ensemble learning method that builds a model by iteratively fitting
    SequentialPathBoost models on different subsets of the data, partitioned by anchor nodes.
    It is designed for graph-based data where paths originating from specified anchor nodes
    are used as features.

    The model trains a separate `SequentialPathBoost` instance for each unique anchor node
    label provided. Predictions are then aggregated (averaged) from these individual models.
    It supports parallel training of the `SequentialPathBoost` models across multiple cores.

    Parameters
    ----------
    n_iter : int, default=100
        The number of boosting iterations to perform for each `SequentialPathBoost` model.
    max_path_length : int, default=10
        The maximum length of paths to consider as features.
    learning_rate : float, default=0.1
        The learning rate shrinks the contribution of each base learner in the `SequentialPathBoost` model.
    m_stops : list[int], default=None
        A list of iteration numbers at which to stop boosting for specific models.
        Currently, this parameter is validated but not fully implemented in the core logic.
    BaseLearnerClass : type, default=sklearn.tree.DecisionTreeRegressor
        The class of the base learner to be used within each boosting iteration in the `SequentialPathBoost` model.
        Must implement the `BaseLearnerClassInterface`.
    kwargs_for_base_learner : dict, default=None
        Keyword arguments to be passed to the constructor of the `BaseLearnerClass`.
    SelectorClass : type, default=sklearn.tree.DecisionTreeRegressor
        The class of the feature selector used to identify the best paths in each iteration.
        Must implement the `SelectorClassInterface`.
    kwargs_for_selector : dict, default=None
        Keyword arguments to be passed to the constructor of the `SelectorClass`.
    parameters_variable_importance : dict, default=None
        Parameters for computing variable importance. If None, variable importance is not computed.
        Expected keys include 'criterion' = 'absolute' or 'relative', 'error_used' = 'mse' or 'mae', 'use_correlation' = True or False, 'normalize' = True or False.
    replace_nan_with : any, default=np.nan
        Value used to replace NaN values encountered during feature generation. It is needed for some base learners like linear models who can not deal with NaN values.
    verbose : bool, default=False
        If True, prints progress messages during fitting.
    n_of_cores : int, default=1
        The number of CPU cores to use for parallel training of `SequentialPathBoost` models.
        If 1, training is sequential.
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "n_iter": [numbers.Integral],
        "max_path_length": [numbers.Integral],
        "learning_rate": [numbers.Integral, numbers.Real],
        "target_error": [numbers.Real, None],
        "base_learner_kwargs": [dict, None],
        "BaseLearnerClass": [type],
        "SelectorClass": [type],
        "kwargs_for_selector": [dict, None],
        "eval_set": [list[tuple[list[nx.Graph], Iterable]], None],
        "list_anchor_nodes_labels": [list[tuple]],
        "X": [list[nx.Graph]],
        "y": [Iterable],
        "anchor_nodes_label_name": [str],
        "verbose": [bool],
        "n_of_cores": [numbers.Integral],
        "parameters_variable_importance": [dict, None]
    }

    def __init__(self, n_iter=100,
                 patience: int | None = None,
                 target_error: float | None = None,
                 max_path_length=10,
                 learning_rate=0.1,
                 m_stops: list[int] = None,
                 BaseLearnerClass=DecisionTreeRegressor,
                 kwargs_for_base_learner=None,
                 SelectorClass=DecisionTreeRegressor,
                 kwargs_for_selector=None,
                 parameters_variable_importance=None,
                 replace_nan_with=np.nan,
                 verbose: bool = False, n_of_cores: int = 1):

        self.n_iter: int = n_iter
        self.patience: int = patience
        self.target_error: float | None = target_error
        self.m_stops: list[int] = m_stops
        self.max_path_length: int = max_path_length
        self.learning_rate: float = learning_rate
        self.BaseLearnerClass: type[BaseLearnerClassInterface] = BaseLearnerClass
        self.verbose: bool = verbose
        self.n_of_cores = n_of_cores
        self.kwargs_for_base_learner: dict = kwargs_for_base_learner
        self.SelectorClass: type[SelectorClassInterface] = SelectorClass
        self.kwargs_for_selector: dict = kwargs_for_selector
        self.replace_nan_with = replace_nan_with
        self.parameters_variable_importance = parameters_variable_importance

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: list[nx.Graph], y: Iterable, anchor_nodes_label_name: str, list_anchor_nodes_labels: list[tuple],
            eval_set: list[tuple[list[nx.Graph], Iterable]] | None = None):
        """
        Fits the PathBoost model to the training data.

        This method trains a `SequentialPathBoost` model for each unique anchor node label.
        The training data `X` and `y` are partitioned based on `list_anchor_nodes_labels`
        and `anchor_nodes_label_name`. Each partition is used to train a corresponding
        `SequentialPathBoost` model. If `n_of_cores` is greater than 1, these models
        are trained in parallel.

        The method also handles the initialization of variable importance computation
        if `parameters_variable_importance` is set. After training, it computes
        the overall training Mean Squared Error (MSE) and, if `eval_set` is provided,
        the MSE for each evaluation set.

        Parameters
        ----------
        X : list[nx.Graph]
            A list of NetworkX graph objects representing the training input samples.
        y : Iterable
            The target values (real numbers in regression) corresponding to `X`.
            Must be array-like of shape (n_samples,) or (n_samples, n_outputs).
        anchor_nodes_label_name : str
            The name of the node attribute in the graphs that identifies the attribute used to identify the anchor nodes.
            e.g. if the anchor nodes are defined by the atomic number, this should be "feature_atomic_number".
        list_anchor_nodes_labels : list[tuple]
            A list of unique labels for the anchor nodes. The data will be partitioned
            based on these labels, and a separate `SequentialPathBoost` model will be
            trained for each.
        eval_set : list[tuple[list[nx.Graph], Iterable]] | None, default=None
            A list of (X_eval, y_eval) tuples for monitoring the model's performance
            on one or more evaluation sets during training.

        Returns
        -------
        self : object
            The fitted PathBoost estimator.
        """
        # Configure logging based on verbose flag
        if self.verbose:
            logging.getLogger('path_boost').setLevel(logging.INFO)
            if not logging.getLogger('path_boost').handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(message)s'))
                logging.getLogger('path_boost').addHandler(handler)

        self._default_kwargs_for_base_learner = {'max_depth': 3, 'random_state': 0,
                                                 'splitter': 'best', 'criterion': "squared_error"}

        self._default_kwargs_for_selector = {'max_depth': 1, 'random_state': 0, 'splitter': 'best',
                                             'criterion': "squared_error"}

        self.anchor_nodes_label_name_ = anchor_nodes_label_name
        self.list_anchor_nodes_labels_ = list_anchor_nodes_labels

        X, y = self._validate_data(X=X, y=y, list_anchor_nodes_labels=list_anchor_nodes_labels, eval_set=eval_set,
                                   m_stops=self.m_stops, name_of_label_attribute=anchor_nodes_label_name,
                                   parameters_variable_importance=self.parameters_variable_importance,
                                   patience=self.patience)

        # if variable importance is used, we need all the sub models to not normalize the data and eventually remember to normalize later
        if self.parameters_variable_importance is not None:
            self.normalize_path_importance_: bool = self.parameters_variable_importance.get('normalize', False)
            self.parameters_variable_importance['normalize'] = False

        self.is_fitted_ = True

        # divide the training dataset by metal center
        indexes_of_train_graphs_for_each_anchor_label: list[list[int]] = wbu.split_dataset_by_metal_centers(
            graphs_list=X,
            anchor_nodes_label_name=self.anchor_nodes_label_name_,
            anchor_nodes=self.list_anchor_nodes_labels_)

        train_datasets_for_each_anchor_label = []
        train_labels_for_each_anchor_label = []

        self.models_list_: list[SequentialPathBoost] = []

        m_stops_counter = 0
        # create a train dataset and model
        for i, _ in enumerate(self.list_anchor_nodes_labels_):
            train_indexes = indexes_of_train_graphs_for_each_anchor_label[i]
            train_dataset = [X[index] for index in train_indexes]
            train_labels = [y[index] for index in train_indexes]
            train_datasets_for_each_anchor_label.append(train_dataset)
            train_labels_for_each_anchor_label.append(train_labels)
            if len(train_dataset) != 0:
                n_iter = self.n_iter
                # needed to be done to distinguish the case when we are given an m_stops for each anchor node or when we are given a m_stop for each trained model
                if self.m_stops is not None:
                    if len(self.m_stops) == len(self.list_anchor_nodes_labels_):
                        n_iter = self.m_stops[i]
                    else:
                        n_iter = self.m_stops[m_stops_counter]
                        m_stops_counter += 1

                self.models_list_.append(
                    SequentialPathBoost(n_iter=n_iter,
                                        patience=self.patience,
                                        target_error=self.target_error,
                                        max_path_length=self.max_path_length,
                                        learning_rate=self.learning_rate,
                                        BaseLearnerClass=self.BaseLearnerClass,
                                        SelectorClass=self.SelectorClass,
                                        kwargs_for_base_learner=self.kwargs_for_base_learner,
                                        kwargs_for_selector=self.kwargs_for_selector,
                                        parameters_variable_importance=self.parameters_variable_importance,
                                        replace_nan_with=self.replace_nan_with,
                                        verbose=self.verbose)
                )

            else:
                # if there is no training data, we will append None to the list of models
                self.models_list_.append(None)

        # parallelization
        # We will use the `wbu.train_pattern_boosting` function to train the model in parallel.
        input_for_parallelization = list(zip(self.models_list_, train_datasets_for_each_anchor_label,
                                             train_labels_for_each_anchor_label,
                                             self.list_anchor_nodes_labels_,
                                             [anchor_nodes_label_name for _ in
                                              range(len(self.list_anchor_nodes_labels_))]))

        number_of_effective_trained_models: int = sum(1 for model in self.models_list_ if model is not None)
        number_of_cores_used = min(mp.cpu_count(), self.n_of_cores, number_of_effective_trained_models)
        if number_of_cores_used <= 1:
            path_boosting_models = []
            # Set up iterator with optional progress bar for sequential training
            iterator = range(len(input_for_parallelization))
            if self.verbose and TQDM_AVAILABLE:
                iterator = tqdm(iterator, desc="Training anchor models", unit="model",
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            for i in iterator:
                path_boosting_models.append(wbu.train_pattern_boosting(input_for_parallelization[i]))

        else:

            with mp.get_context("spawn").Pool(number_of_cores_used) as pool:
                path_boosting_models = pool.map(wbu.train_pattern_boosting, input_for_parallelization)

        self.models_list_ = path_boosting_models
        self.train_mse_ = self._compute_train_mse(
            number_of_observations_for_each_model=[len(dataset) for dataset in train_datasets_for_each_anchor_label])

        if eval_set is not None:
            self.mse_eval_set_ = []
            for eval_tuple in eval_set:
                self.mse_eval_set_.append(self.evaluate(X=eval_tuple[0], y=eval_tuple[1]))

        if self.parameters_variable_importance is not None:
            self.compute_variable_importance()

        # `fit` should always return `self`
        return self

    def compute_variable_importance(self):
        self.parameters_variable_importance['normalize'] = self.normalize_path_importance_

        self.variable_importance_ = VariableImportance_ForSequentialPathBoost(
            **self.parameters_variable_importance, ).combine_variable_importance_from_list_of_sequential_models(
            sequential_models=self.models_list_, )

    def _compute_train_mse(self, number_of_observations_for_each_model: list[int]):
        train_mse = np.zeros(self.n_iter)
        for i, smc_model in enumerate(self.models_list_):
            if smc_model is not None:
                train_mse += np.array(smc_model.train_mse_) * number_of_observations_for_each_model[i]
        train_mse = train_mse / sum(number_of_observations_for_each_model)
        return train_mse

    def predict(self, X: List[nx.Graph]) -> List[float]:
        """
        Predict target values for the input graphs.

        This method partitions the input graphs by anchor nodes, generates
        predictions using the corresponding trained SequentialPathBoost models,
        and averages predictions across models for graphs with multiple anchor types.

        Parameters
        ----------
        X : List[nx.Graph]
            A list of NetworkX graph objects for which to make predictions.

        Returns
        -------
        predictions : List[float]
            Predicted target values for each input graph.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the model has not been fitted yet.

        Examples
        --------
        >>> predictions = model.predict(X_test)  # doctest: +SKIP
        >>> print(f"Mean prediction: {np.mean(predictions):.3f}")  # doctest: +SKIP
        """
        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X=X)

        # divide the input by the anchor node
        indexes_of_graphs_for_each_anchor_label: list[list[int]] = wbu.split_dataset_by_metal_centers(
            graphs_list=X,
            anchor_nodes_label_name=self.anchor_nodes_label_name_,
            anchor_nodes=self.list_anchor_nodes_labels_)

        # create the dataset for each anchor node
        datasets_for_each_anchor_label = []
        for i, _ in enumerate(self.list_anchor_nodes_labels_):
            indexes = indexes_of_graphs_for_each_anchor_label[i]
            dataset = [X[index] for index in indexes]
            datasets_for_each_anchor_label.append(dataset)

        number_of_effective_trained_models: int = sum(1 for model in self.models_list_ if model is not None)
        number_of_dataset_to_be_predicted = sum(1 for dataset in datasets_for_each_anchor_label if len(dataset) != 0)
        number_of_cores_used = min(mp.cpu_count(), self.n_of_cores, number_of_dataset_to_be_predicted,
                                   number_of_effective_trained_models)

        if number_of_cores_used <= 1:
            predictions_for_each_anchor_node = []
            for i in range(len(datasets_for_each_anchor_label)):
                if self.models_list_[i] is not None:
                    predictions = wbu.parallel_predict(
                        input_from_parallelization=(self.models_list_[i], datasets_for_each_anchor_label[i]))
                    predictions_for_each_anchor_node.append(predictions)
                else:
                    predictions_for_each_anchor_node.append(None)
        else:
            input_for_parallelization = list(zip(self.models_list_, datasets_for_each_anchor_label))
            with mp.get_context("spawn").Pool(self.n_of_cores) as pool:
                predictions_for_each_anchor_node = pool.map(wbu.parallel_predict, input_for_parallelization)

        # create a matrix (list of lists) where the columns refer to the anchor nodes and the rows to the graphs
        predictions_for_each_anchor_node_padded_with_none = [[None for _ in range(len(X))] for _ in
                                                             range(len(self.list_anchor_nodes_labels_))]
        for anchor_node_number in range(len(self.list_anchor_nodes_labels_)):
            for i in range(len(indexes_of_graphs_for_each_anchor_label[anchor_node_number])):
                graph_number = indexes_of_graphs_for_each_anchor_label[anchor_node_number][i]
                predictions_for_each_anchor_node_padded_with_none[anchor_node_number][graph_number] = \
                    predictions_for_each_anchor_node[anchor_node_number][i]

        # Transpose the list of lists, filling missing values with None
        transposed_list = list(
            map(list, itertools.zip_longest(*predictions_for_each_anchor_node_padded_with_none, fillvalue=None)))

        # Calculate the average of each row, ignoring None values
        predictions = []
        for sublist in transposed_list:
            if len(sublist) > 0:
                non_none_values = [x for x in sublist if x is not None]
                if len(non_none_values) > 0:
                    avg = np.mean(non_none_values)
                else:
                    avg = 0
            else:
                avg = 0
            predictions.append(avg)

        predictions = [x if x is not None and not np.isnan(x) else 0 for x in predictions]

        return predictions

    def predict_step_by_step(self, X: list[nx.Graph]) -> list[list[float]]:
        """
        Predicts the target values for the input graphs step by step, returning the predictions at each iteration.

        This method divides the input graphs by anchor nodes, generates datasets for each anchor node, and then
        uses the trained models to predict the target values for each dataset. The predictions are made iteratively,
        and the method returns the predictions at each iteration.

        Parameters
        ----------
        X : list[nx.Graph]
            A list of networkx graph objects to be used for prediction.

        Returns
        -------
        list[list[float]]
            A list of lists where each inner list contains the predictions for the input graphs at a specific iteration.
            The outer list contains the predictions for all iterations.
        """

        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X=X)

        # divide the input by the anchor node
        indexes_of_graphs_for_each_anchor_label: list[list[int]] = wbu.split_dataset_by_metal_centers(
            graphs_list=X,
            anchor_nodes_label_name=self.anchor_nodes_label_name_,
            anchor_nodes=self.list_anchor_nodes_labels_)

        # create the dataset for each anchor node
        datasets_for_each_anchor_label = []
        for i, _ in enumerate(self.list_anchor_nodes_labels_):
            indexes = indexes_of_graphs_for_each_anchor_label[i]
            dataset = [X[index] for index in indexes]
            datasets_for_each_anchor_label.append(dataset)

        number_of_effective_trained_models: int = sum(1 for model in self.models_list_ if model is not None)
        number_of_dataset_to_be_predicted = sum(1 for dataset in datasets_for_each_anchor_label if len(dataset) != 0)
        number_of_cores_used = min(mp.cpu_count(), self.n_of_cores, number_of_dataset_to_be_predicted,
                                   number_of_effective_trained_models)

        # get the step by step predictions for each anchor node
        if number_of_cores_used <= 1:
            step_by_step_predictions_for_each_anchor_node: list[list[list[numbers.Number]]] = []
            for i in range(len(datasets_for_each_anchor_label)):
                if self.models_list_[i] is not None:
                    predictions_step_by_step = wbu.parallel_predict_step_by_step(
                        (self.models_list_[i], datasets_for_each_anchor_label[i]))
                    step_by_step_predictions_for_each_anchor_node.append(predictions_step_by_step)
                else:
                    step_by_step_predictions_for_each_anchor_node.append(None)
        else:
            input_for_parallelization = list(zip(self.models_list_, datasets_for_each_anchor_label))
            with mp.get_context("spawn").Pool(self.n_of_cores) as pool:
                step_by_step_predictions_for_each_anchor_node: list[list[list[numbers.Number]]] = pool.map(
                    wbu.parallel_predict_step_by_step, input_for_parallelization)

        # create a matrix for each iteration (list of lists) where the columns refer to the anchor nodes and the rows to the graphs
        iterations_predictions_for_each_anchor_node_padded_with_none = []
        for iteration in range(self.n_iter):

            predictions_for_each_anchor_node_padded_with_none = [[None for _ in range(len(X))] for _ in
                                                                 range(len(self.list_anchor_nodes_labels_))]

            for anchor_node_number in range(len(self.list_anchor_nodes_labels_)):
                for i in range(len(indexes_of_graphs_for_each_anchor_label[anchor_node_number])):
                    graph_number = indexes_of_graphs_for_each_anchor_label[anchor_node_number][i]
                    predictions_for_each_anchor_node_padded_with_none[anchor_node_number][graph_number] = \
                        step_by_step_predictions_for_each_anchor_node[anchor_node_number][iteration][i]

            iterations_predictions_for_each_anchor_node_padded_with_none.append(
                predictions_for_each_anchor_node_padded_with_none)

        transposed_iteration_predictions = []
        for iteration in range(self.n_iter):
            # Transpose the list of lists, filling missing values with None
            transposed_list = list(
                map(list,
                    itertools.zip_longest(*iterations_predictions_for_each_anchor_node_padded_with_none[iteration],
                                          fillvalue=None)))

            transposed_iteration_predictions.append(transposed_list)

        # Calculate the average of each row, ignoring None values
        predictions_step_by_step = []
        for iteration in range(self.n_iter):
            averages = []
            for sublist in transposed_iteration_predictions[iteration]:
                if len(sublist) > 0:
                    non_none_values = [x for x in sublist if x is not None]
                    if len(non_none_values) > 0:
                        avg = np.mean(non_none_values)
                    else:
                        avg = 0
                else:
                    avg = 0
                averages.append(avg)
            averages = [x if x is not None and not np.isnan(x) else 0 for x in averages]
            predictions_step_by_step.append(averages)

        return predictions_step_by_step

    def _merge_values_from_single_path_boost(self, len_X: int, indexes_of_graphs_for_each_anchor_label: list[list[int]],
                                             values_for_each_anchor_node: list[list[float]]):
        """
        This method is used to merge (average) the values (predictions) from a SingleMetalCenterPathBoost instance into the current instance of PathBoost
        """

        averaged_values = [0 for _ in range(len_X)]
        counter = [0 for _ in range(len_X)]
        for graph_number in range(len_X):
            for anchor_node_number in range(len(self.list_anchor_nodes_labels_)):
                if graph_number in indexes_of_graphs_for_each_anchor_label[anchor_node_number]:
                    graph_position_in_sub_dataset = indexes_of_graphs_for_each_anchor_label[anchor_node_number].index(
                        graph_number)
                    averaged_values[graph_number] += values_for_each_anchor_node[anchor_node_number][
                        graph_position_in_sub_dataset]
                    counter[graph_number] += 1

        averaged_values = np.divide(averaged_values, counter, out=np.zeros_like(averaged_values), where=counter != 0)

        return averaged_values

    def evaluate(self, X: list[nx.Graph], y: Iterable) -> list[float]:

        # it returns the evolution of the mse with increasing number of iterations
        predictions = self.predict_step_by_step(X)
        evolution_mse = []
        for prediction in predictions:
            mse = mean_squared_error(y_true=y, y_pred=prediction)
            evolution_mse.append(mse)
        return evolution_mse

    def plot_training_and_eval_errors(self, skip_first_n_iterations: int | bool = True, plot_eval_sets_error=True,
                                      show=True, save=False, save_path: str | None = None):
        """
        Plots the training and evaluation set errors over iterations.
        """
        if hasattr(self, 'mse_eval_set_') and plot_eval_sets_error is True:
            eval_sets_mse = self.mse_eval_set_
        else:
            eval_sets_mse = None
        plot_training_and_eval_errors(learning_rate=self.learning_rate, train_mse=self.train_mse_,
                                      mse_eval_set=eval_sets_mse, skip_first_n_iterations=skip_first_n_iterations,
                                      show=show, save=save, save_path=save_path)

    def plot_variable_importance(self, top_n_features: int | None = None, show: bool = True):
        if self.parameters_variable_importance is None:
            raise ValueError(
                "Variable importance is not computed. Please set parameters_variable_importance in the constructor.")
        plot_variable_importance_utils(variable_importance=self.variable_importance_,
                                       parameters_variable_importance=self.parameters_variable_importance,
                                       top_n=top_n_features, show=show)

    def score(self, X, y, sample_weight=None):
        # This method is used to evaluate the model on the given data.
        # It is defined in the `RegressorMixin` class.
        # It allows to:
        # - evaluate the model on the given data
        # - return the score
        mse_evolution = self.evaluate(X=X, y=y)
        best_mse = mse_evolution[-1]
        return - best_mse

    def _validate_data(
            self,
            X="no_validation",
            y="no_validation",
            reset=True,
            validate_separately=False,
            **check_params,
    ):

        util_validate_data(model=self, X=X, y=y, reset=reset, validate_separately=validate_separately, **check_params)

        if not np.array_equal(y, "no_validation"):
            validate_data(self,
                          X="no_validation",
                          y=y,
                          reset=reset,
                          validate_separately=validate_separately
                          )

        if not np.array_equal(X, "no_validation") and not np.array_equal(y, "no_validation"):
            return X, y
        elif not np.array_equal(X, "no_validation"):
            return X
        elif not np.array_equal(y, "no_validation"):
            return y

    def get_final_eval_set_mse(self):
        """
        Returns the evaluation set MSE if it was computed during fitting.
        """
        if hasattr(self, 'mse_eval_set_'):
            final_eval_set_mse= []
            for mse in self.mse_eval_set_:
                final_eval_set_mse.append(mse[-1])
            return final_eval_set_mse
        else:
            raise AttributeError("Evaluation set MSE is not available. Please fit the model with eval_set.")

    def save(self, filepath: str) -> None:
        """
        Save the fitted model to a file.

        This method serializes the entire PathBoost model, including all trained
        SequentialPathBoost sub-models, to a file using joblib. The saved file
        includes metadata such as the package version and training parameters.

        Parameters
        ----------
        filepath : str
            The path where the model should be saved. The file extension
            `.joblib` is recommended but not required.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = PathBoost(n_iter=50)  # doctest: +SKIP
        >>> model.fit(X_train, y_train, anchor_nodes_label_name='atomic_number',
        ...           list_anchor_nodes_labels=[1, 6, 7, 8])  # doctest: +SKIP
        >>> model.save('my_model.joblib')  # doctest: +SKIP

        See Also
        --------
        load : Load a saved model from file.
        """
        import joblib
        from datetime import datetime
        from ._version import __version__

        check_is_fitted(self)

        # Create metadata dictionary
        metadata = {
            'version': __version__,
            'saved_at': datetime.now().isoformat(),
            'n_iter': self.n_iter,
            'max_path_length': self.max_path_length,
            'learning_rate': self.learning_rate,
            'anchor_nodes_label_name': self.anchor_nodes_label_name_,
            'list_anchor_nodes_labels': self.list_anchor_nodes_labels_,
            'n_models': len(self.models_list_),
        }

        # Package model and metadata
        save_dict = {
            'model': self,
            'metadata': metadata,
        }

        joblib.dump(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'PathBoost':
        """
        Load a saved PathBoost model from a file.

        This class method deserializes a PathBoost model that was previously
        saved using the `save` method. It restores the complete model state
        including all trained sub-models and parameters.

        Parameters
        ----------
        filepath : str
            The path to the saved model file.

        Returns
        -------
        PathBoost
            The loaded PathBoost model, ready for prediction.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file does not contain a valid PathBoost model.

        Warns
        -----
        UserWarning
            If the model was saved with a different version of the package.

        Examples
        --------
        >>> model = PathBoost.load('my_model.joblib')  # doctest: +SKIP
        >>> predictions = model.predict(X_test)  # doctest: +SKIP

        See Also
        --------
        save : Save a fitted model to file.

        Notes
        -----
        Models saved with older versions of the package may not be fully
        compatible. A warning will be issued if version mismatch is detected.
        """
        import joblib
        from ._version import __version__

        save_dict = joblib.load(filepath)

        # Validate the loaded object
        if not isinstance(save_dict, dict) or 'model' not in save_dict:
            raise ValueError(
                f"Invalid model file. Expected a PathBoost save file, but got {type(save_dict)}"
            )

        model = save_dict['model']
        metadata = save_dict.get('metadata', {})

        # Check version compatibility
        saved_version = metadata.get('version', 'unknown')
        if saved_version != __version__:
            warnings.warn(
                f"Model was saved with version {saved_version}, but current version is {__version__}. "
                "Some features may not work correctly.",
                UserWarning
            )

        logger.info(f"Model loaded from {filepath} (saved: {metadata.get('saved_at', 'unknown')})")
        return model

    def predict_with_uncertainty(
        self,
        X: List[nx.Graph],
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict target values with uncertainty estimates using anchor node bootstrap.

        This method leverages the natural ensemble structure of PathBoost, where
        separate models are trained for each anchor node type. Instead of simply
        averaging predictions across anchor models, this method computes the
        variance across individual model predictions to estimate uncertainty.

        Parameters
        ----------
        X : List[nx.Graph]
            A list of NetworkX graph objects for which to make predictions.
        confidence : float, default=0.95
            The confidence level for the prediction intervals. Must be between
            0 and 1. Common values are 0.90, 0.95, and 0.99.

        Returns
        -------
        predictions : np.ndarray
            Point estimates (mean predictions) for each input graph.
            Shape: (n_samples,)
        lower_bounds : np.ndarray
            Lower bounds of the confidence interval for each prediction.
            Shape: (n_samples,)
        upper_bounds : np.ndarray
            Upper bounds of the confidence interval for each prediction.
            Shape: (n_samples,)

        Raises
        ------
        ValueError
            If confidence is not between 0 and 1.
            If fewer than 2 anchor node models were trained (uncertainty requires
            multiple models for bootstrap estimation).

        Notes
        -----
        The uncertainty estimates are based on the variance across anchor node
        models. This approach is valid when:
        - Multiple anchor node types were used during training
        - Each graph contains nodes from multiple anchor types

        For graphs that only match a single anchor type, the uncertainty
        estimate will be based on fewer models and may be less reliable.

        The confidence intervals assume approximate normality of the prediction
        distribution across anchor models. For small numbers of anchor types,
        this assumption may not hold exactly.

        Examples
        --------
        >>> model = PathBoost(n_iter=50)  # doctest: +SKIP
        >>> model.fit(X_train, y_train, anchor_nodes_label_name='atomic_number',
        ...           list_anchor_nodes_labels=[6, 7, 8, 9])  # doctest: +SKIP
        >>> predictions, lower, upper = model.predict_with_uncertainty(X_test)  # doctest: +SKIP
        >>> for i in range(5):  # doctest: +SKIP
        ...     print(f"Pred: {predictions[i]:.2f}, CI: [{lower[i]:.2f}, {upper[i]:.2f}]")

        See Also
        --------
        predict : Standard prediction without uncertainty estimates.
        """
        from scipy import stats

        # Validate inputs
        check_is_fitted(self)

        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

        n_effective_models = sum(1 for m in self.models_list_ if m is not None)
        if n_effective_models < 2:
            raise ValueError(
                f"predict_with_uncertainty requires at least 2 trained anchor models, "
                f"but only {n_effective_models} model(s) were trained. "
                "Use predict() for standard predictions."
            )

        X = self._validate_data(X=X)

        # Get predictions from each anchor model separately
        indexes_of_graphs_for_each_anchor_label: List[List[int]] = wbu.split_dataset_by_metal_centers(
            graphs_list=X,
            anchor_nodes_label_name=self.anchor_nodes_label_name_,
            anchor_nodes=self.list_anchor_nodes_labels_
        )

        # Create datasets for each anchor node
        datasets_for_each_anchor_label = []
        for i, _ in enumerate(self.list_anchor_nodes_labels_):
            indexes = indexes_of_graphs_for_each_anchor_label[i]
            dataset = [X[index] for index in indexes]
            datasets_for_each_anchor_label.append(dataset)

        # Get predictions from each model
        predictions_matrix = np.full((len(X), len(self.list_anchor_nodes_labels_)), np.nan)

        for anchor_idx, (model, dataset) in enumerate(zip(self.models_list_, datasets_for_each_anchor_label)):
            if model is not None and len(dataset) > 0:
                preds = model.predict(dataset)
                graph_indices = indexes_of_graphs_for_each_anchor_label[anchor_idx]
                for local_idx, global_idx in enumerate(graph_indices):
                    predictions_matrix[global_idx, anchor_idx] = preds[local_idx]

        # Compute statistics across anchor models
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Mean prediction (ignoring NaN)
            predictions = np.nanmean(predictions_matrix, axis=1)
            # Standard deviation across models
            std_devs = np.nanstd(predictions_matrix, axis=1, ddof=1)
            # Count of models contributing to each prediction
            n_models_per_sample = np.sum(~np.isnan(predictions_matrix), axis=1)

        # Replace NaN predictions with 0 (same as regular predict)
        predictions = np.where(np.isnan(predictions), 0, predictions)

        # Compute confidence intervals
        # Use t-distribution for small sample sizes
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha / 2)  # Two-tailed

        # Standard error of the mean
        standard_errors = std_devs / np.sqrt(np.maximum(n_models_per_sample, 1))

        # For samples with only 1 model, set uncertainty to 0 (no information)
        standard_errors = np.where(n_models_per_sample < 2, 0, standard_errors)

        lower_bounds = predictions - z_score * standard_errors
        upper_bounds = predictions + z_score * standard_errors

        return predictions, lower_bounds, upper_bounds


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator = check_estimator(PathBoost())
