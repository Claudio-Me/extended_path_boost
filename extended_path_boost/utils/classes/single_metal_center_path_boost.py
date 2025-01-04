import networkx as nx
from sklearn.base import BaseEstimator
import numpy as np
from .extended_boosting_matrix import ExtendedBoostingMatrix


class SingleMetalCenterPathBoost(BaseEstimator):
    def __init__(self, n_iter=100, max_path_length=10, learning_rate=0.1, base_learner="Tree",
                 selector="tree", base_learner_parameters=None):
        self.n_iter = n_iter
        self.max_path_length = max_path_length
        self.learning_rate = learning_rate
        self.base_learner = base_learner
        self.selector = selector
        self.base_learner_parameters = base_learner_parameters

    def fit(self, X: list[nx.Graph],y: np.array,label_metal_center, eval_set=None):

        self.is_fitted_ = True
        self._inizialize_path_boosting(label_metal_center= label_metal_center)

        # `fit` should always return `self`
        return self


    def predict(self, dataset):
        return

    def score(self, dataset):
        return

    def _inizialize_path_boosting(self, label_metal_center):
        self.train_ebm_dataframe = ExtendedBoostingMatrix.create_extend_boosting_matrix_for(
            selected_paths=selected_paths, list_graphs_nx=train_data, settings=self.settings)