"""
This is a module to be used as a reference for building other modules
"""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
import networkx as nx
from .utils.classes.single_metal_center_path_boost import SingleMetalCenterPathBoost


def train_pattern_boosting(input_from_parallelization: tuple):
    model: SingleMetalCenterPathBoost = input_from_parallelization[0]
    train_dataset = input_from_parallelization[1]
    test_dataset = input_from_parallelization[2]
    global_labels_variance = input_from_parallelization[3]
    model.fit(train_dataset, test_dataset,
                                    global_train_labels_variance=global_labels_variance)
    return model


class PathBoost(BaseEstimator):
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
        "base_learner": [str],
        "selector": [str],
        "base_learner_parameters": [dict, None],
    }

    def __init__(self, n_iter=100, max_path_length=10, learning_rate=0.1, base_learner="Tree",
                 selector="tree", base_learner_parameters=None):
        self.n_iter = n_iter
        self.max_path_length = max_path_length
        self.learning_rate = learning_rate
        self.base_learner = base_learner
        self.selector = selector
        self.base_learner_parameters = base_learner_parameters

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, eval_set=None):
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
        # `_validate_data` is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.
        X, y = self._validate_data(X, y)
        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def predict(self, X):
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
        return np.ones(len(X), dtype=np.int64)

    def _validate_data(
            self,
            X="no_validation",
            y="no_validation",
            reset=True,
            validate_separately=False,
            cast_to_ndarray=True,
            **check_params,
    ):
        """
        Validate input data and set or check the `n_features_in_` attribute.
        We use the `_validate_data` method implemented in the super class `BaseEstimator`. We only personalize the check on the X's because we expect a list of networkx graphs instead of a numpy array.
        """
        # We use the `_validate_data` method to validate the input data.
        # This method is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.
        if not np.array_equal(X, "no_validation"):
            assert isinstance(X, list) and all(isinstance(item, nx.Graph) for item in X)
        if not np.array_equal(y, "no_validation"):
            super()._validate_data(
                X="no_validation",
                y=y,
                reset=reset,
                validate_separately=validate_separately,
                cast_to_ndarray=cast_to_ndarray,
                **check_params,
            )

        if not np.array_equal(X, "no_validation") and not np.array_equal(y, "no_validation"):
            return X, y
        elif not np.array_equal(X, "no_validation"):
            return X
        elif not np.array_equal(y, "no_validation"):
            return y

    @staticmethod
    def _split_dataset_by_metal_centers(graphs_list: list[nx.Graph], anchor_nodes_label_name: str,
                                        anchor_nodes: list):
        """
            Splits a list of graphs into subgroups based on anchor node labels.

            This static method takes a list of graphs, an anchor nodes label name,
            and a list of anchor nodes. It iterates through each graph and identifies
            nodes labeled with anchor node labels. It then organizes the indices of
            graphs where such anchor nodes are found into corresponding subgroups.

            Args:
                graphs_list (list[nx.Graph]): A list of networkx Graph objects to be processed.
                anchor_nodes_label_name (str): The name of the attribute used to identify anchor nodes in the graphs.
                anchor_nodes (list): A list of anchor nodes to be used as a reference for grouping.

            Returns:
                list[list[int]]: A list containing sublists of indices corresponding to the grouping
                                 of graphs based on the presence of the anchor nodes.
        """

        indices_list = [[] for _ in range(len(anchor_nodes))]
        for i, graph in enumerate(graphs_list):
            for node, attributes in graph.nodes(data=True):
                if attributes.get(anchor_nodes_label_name) in anchor_nodes:
                    index_in_anchor_nodes = anchor_nodes.index(attributes.get(anchor_nodes_label_name))
                    indices_list[index_in_anchor_nodes].append(i)

        return indices_list


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator = check_estimator(PathBoost())
