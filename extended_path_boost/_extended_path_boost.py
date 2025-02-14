"""
This is a module to be used as a reference for building other modules
"""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause


import os

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

from .utils.classes.single_metal_center_path_boost import SingleMetalCenterPathBoost
from .utils import wrapper_path_boost_utils as wbu
from .utils.classes.interfaces.interface_base_learner import BaseLearnerClassInterface
from .utils.classes.interfaces.interface_selector import SelectorClassInterface
from typing import Iterable
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, _fit_context
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import RegressorMixin
from matplotlib.ticker import MaxNLocator


class PathBoost(BaseEstimator, RegressorMixin):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstration of how to pass and store parameters.

    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "n_iter": [int],
        "max_path_length": [int],
        "learning_rate": [float],
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
        "n_of_cores": [int],
    }

    def __init__(self, n_iter=100, max_path_length=10, learning_rate=0.1, BaseLearnerClass=DecisionTreeRegressor,
                 kwargs_for_base_learner=None, SelectorClass=DecisionTreeRegressor, kwargs_for_selector=None,
                 verbose: bool = False, n_of_cores: int = 1):
        if kwargs_for_base_learner is None:
            kwargs_for_base_learner = {}
        self.n_iter = n_iter
        self.max_path_length = max_path_length
        self.learning_rate = learning_rate
        self.BaseLearnerClass = BaseLearnerClass
        self.verbose = verbose
        self.n_of_cores = n_of_cores

        # very basic logic in the __init__ it is just to have a more clean code, it set kwargs with default dictionaries if no other input is given, it is possible to move the dictionaries as default parameters
        self.kwargs_for_base_learner = kwargs_for_base_learner
        self.SelectorClass = SelectorClass
        self.kwargs_for_selector = kwargs_for_selector

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: list[nx.Graph], y: Iterable, anchor_nodes_label_name: str, list_anchor_nodes_labels: list[tuple],
            eval_set: list[tuple[list[nx.Graph], Iterable]] | None = None):
        """A reference implementation of a fitting function.

        Parameters
        ----------

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """

        self._default_kwargs_for_base_learner = {'max_depth': 3, 'random_state': 0,
                                                 'splitter': 'best', 'criterion': "squared_error"}

        self._default_kwargs_for_selector = {'max_depth': 1, 'random_state': 0, 'splitter': 'best',
                                             'criterion': "squared_error"}

        X, y = self._validate_data(X, y, list_anchor_nodes_labels=list_anchor_nodes_labels, eval_set=eval_set)

        self.anchor_nodes_label_name_ = anchor_nodes_label_name
        self.list_anchor_nodes_labels_ = list_anchor_nodes_labels

        self.is_fitted_ = True

        # divide the training dataset by metal center
        indexes_of_train_graphs_for_each_anchor_label: list[list[int]] = wbu.split_dataset_by_metal_centers(
            graphs_list=X,
            anchor_nodes_label_name=self.anchor_nodes_label_name_,
            anchor_nodes=self.list_anchor_nodes_labels_)

        train_datasets_for_each_anchor_label = []
        train_labels_for_each_anchor_label = []

        self.models_list_: list[SingleMetalCenterPathBoost] = []

        # create a train dataset and model
        for i, _ in enumerate(self.list_anchor_nodes_labels_):
            train_indexes = indexes_of_train_graphs_for_each_anchor_label[i]
            train_dataset = [X[index] for index in train_indexes]
            train_labels = [y[index] for index in train_indexes]
            train_datasets_for_each_anchor_label.append(train_dataset)
            train_labels_for_each_anchor_label.append(train_labels)
            if len(train_dataset) != 0:
                self.models_list_.append(
                    SingleMetalCenterPathBoost(n_iter=self.n_iter,
                                               max_path_length=self.max_path_length,
                                               learning_rate=self.learning_rate,
                                               BaseLearnerClass=self.BaseLearnerClass,
                                               SelectorClass=self.SelectorClass,
                                               kwargs_for_base_learner=self.kwargs_for_base_learner,
                                               kwargs_for_selector=self.kwargs_for_selector, verbose=self.verbose)
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
            for i in range(len(input_for_parallelization)):
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

        # `fit` should always return `self`
        return self

    def _compute_train_mse(self, number_of_observations_for_each_model: list[int]):
        train_mse = np.zeros(self.n_iter)
        for i, smc_model in enumerate(self.models_list_):
            if smc_model is not None:
                train_mse += np.array(smc_model.train_mse_) * number_of_observations_for_each_model[i]
        train_mse = train_mse / sum(number_of_observations_for_each_model)
        return train_mse

    def predict(self, X: list[nx.Graph]):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
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
        # predictions = [np.mean([x for x in sublist if x is not None]) for sublist in transposed_list]
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

        # predictions = [np.mean([x for x in sublist if x is not None]) if len(sublist) > 0 else 0 for sublist in transposed_list]
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
            # averages = [np.mean([x for x in sublist if x is not None]) if len(sublist) > 0 else 0 for sublist in transposed_iteration_predictions[iteration]]
            averages = [x if x is not None and not np.isnan(x) else 0 for x in averages]
            predictions_step_by_step.append(averages)

        return predictions_step_by_step

    def _merge_values_from_single_path_boost(self, len_X: int, indexes_of_graphs_for_each_anchor_label: list[list[int]],
                                             values_for_each_anchor_node: list[list[float]]):
        """
        This method is used to merge (average) the values (predictions) from a SingleMetalCenterPathBoost instance into the current instance of PathBoost
        """

        predictions = [0 for _ in range(len_X)]
        counter = [0 for _ in range(len_X)]
        for graph_number in range(len_X):
            for anchor_node_number in range(len(self.list_anchor_nodes_labels_)):
                if graph_number in indexes_of_graphs_for_each_anchor_label[anchor_node_number]:
                    graph_position_in_sub_dataset = indexes_of_graphs_for_each_anchor_label[anchor_node_number].index(
                        graph_number)
                    predictions[graph_number] += values_for_each_anchor_node[anchor_node_number][
                        graph_position_in_sub_dataset]
                    counter[graph_number] += 1

        predictions = np.divide(predictions, counter, out=np.zeros_like(predictions), where=counter != 0)

        return predictions

    def evaluate(self, X: list[nx.Graph], y: Iterable) -> list[float]:

        # it returns the evolution of the mse with increasing number of iterations
        predictions = self.predict_step_by_step(X)
        evolution_mse = []
        for prediction in predictions:
            mse = mean_squared_error(y_true=y, y_pred=prediction)
            evolution_mse.append(mse)
        return evolution_mse

    def plot_training_and_eval_errors(self):
        """
        Plots the training and evaluation set errors over iterations.
        """
        # skip_the_first n iterations
        n = int(2 / self.learning_rate)
        if len(self.train_mse_) > n:
            train_mse = self.train_mse_[n:]
        else:
            train_mse = self.train_mse_

        plt.figure(figsize=(12, 6))

        # Plot training errors
        plt.plot(range(n, len(train_mse) + n), train_mse, label='Training Error', marker='.')

        # Plot evaluation set errors if available
        if hasattr(self, 'mse_eval_set_'):
            if len(self.mse_eval_set_[0]) > n:
                eval_set_mse = [self.mse_eval_set_[i][n:] for i in range(len(self.mse_eval_set_))]
            else:
                eval_set_mse = self.mse_eval_set_

            num_iterations = len(eval_set_mse[0])
            num_eval_sets = len(eval_set_mse)
            for eval_set_index in range(num_eval_sets):
                if eval_set_mse[eval_set_index][0] is not None:
                    plt.plot(range(n, num_iterations + n), eval_set_mse[eval_set_index],
                             label=f'Evaluation Set {eval_set_index + 1} Error', marker='.')

        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('Training and Evaluation Set Errors Over Iterations')
        plt.legend()
        plt.grid(True)

        # Ensure x-axis only shows integers
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()

    def score(self, X, y, sample_weight=None):
        # This method is used to evaluate the model on the given data.
        # It is defined in the `RegressorMixin` class.
        # It allows to:
        # - evaluate the model on the given data
        # - return the score
        mse_evolution = self.evaluate(X=X, y=y)
        best_mse = max(mse_evolution)
        return - best_mse

    def _validate_data(
            self,
            X="no_validation",
            y="no_validation",
            reset=True,
            validate_separately=False,
            **check_params,
    ):

        # We use the `_validate_data` method to validate the input data.
        # This method is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data
        if not np.array_equal(X, "no_validation"):
            assert isinstance(X, list) and all(isinstance(item, nx.Graph) for item in X)
        if not np.array_equal(y, "no_validation"):
            assert isinstance(y, Iterable) and all(isinstance(item, numbers.Number) for item in y)

        # check BaseLearnerClass and SelectorClass respects the respective interfaces
        if not issubclass(self.BaseLearnerClass, BaseLearnerClassInterface):
            raise TypeError(f"{self.BaseLearnerClass.__name__} must implement BaseLearnerClassInterface")

        if not issubclass(self.SelectorClass, SelectorClassInterface):
            raise TypeError(f"{self.SelectorClass.__name__} must implement BaseLearnerClassInterface")

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

        # check list_anchor_nodes_labels
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

        # check eval sets
        eval_set = check_params.get('eval_set', None)
        if eval_set is not None:
            assert isinstance(eval_set, Iterable) and all(
                isinstance(eval_tuple, tuple) and len(eval_tuple) == 2 for eval_tuple in eval_set)
            for eval_tuple in eval_set:
                assert isinstance(eval_tuple[0], list) and all(isinstance(item, nx.Graph) for item in eval_tuple[0])
                assert isinstance(eval_tuple[1], Iterable) and all(
                    isinstance(item, numbers.Number) for item in eval_tuple[1])

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


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator = check_estimator(PathBoost())
