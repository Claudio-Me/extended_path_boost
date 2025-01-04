"""This file will just show how to write tests for the template classes."""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose, assert_array_equal
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os

from extended_path_boost import PathBoost


# Authors: scikit-learn-contrib developers
# License: BSD 3 clause


@pytest.fixture
def parameters():
    return {"n_iter": 100, "max_path_length": 10, "learning_rate": 0.1, "base_learner": "Tree",
            "selector": "tree", "base_learner_parameters": None,
            "anchor_nodes": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # first block
                             39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # second block
                             57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,  # lanthanides
                             72, 73, 74, 75, 76, 77, 78, 79, 80,  # third block
                             89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,  # actinides
                             104, 105, 106, 107, 108, 109, 110, 111, 112]  # fourth block}
            }


@pytest.fixture
def nx_graphs()->tuple[list[nx.Graph], np.ndarray]:
    path_to_file = os.path.join('test_datasets', 'uNatQ_nx_graphs_subsample.pkl')
    with open(path_to_file, "rb") as f:
        graphs = pickle.load(f)

    y = np.array([graph.graph['target_tzvp_homo_lumo_gap'] for graph in graphs])

    return graphs, y


def test_dataset_splitting(nx_graphs, parameters):
    """Check the internals and behaviour of `TemplateEstimator`."""
    path_boost = PathBoost()

    # assert est.demo_param == "demo_param"



    path_boost.fit(nx_graphs[0], nx_graphs[1])
    assert hasattr(path_boost, "is_fitted_")

    X = nx_graphs[0]
    y_pred = path_boost.predict(X)
    assert_array_equal(y_pred, np.ones(len(X), dtype=np.int64))
