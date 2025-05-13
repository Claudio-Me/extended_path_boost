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
            "anchor_nodes": [(1,), (2,)], "anchor_nodes_label_name": "label"
            }

@pytest.fixture
def nx_graphs() -> tuple[list[nx.Graph], np.ndarray]:
    """Fixture to create sample graph data and target values for tests."""
    G1 = nx.Graph()
    G1.add_nodes_from([
        (1, {"attr1": 1, "label": 1}),
        (2, {"attr2": 2, "label": 2}),
        (3, {"attr3": 3, "label": 2}),
        (4, {"attr4": 4, "label": 3}),
        (5, {"attr5": 5, "label": 4})
    ])
    G1.add_edges_from([
        (1, 2, {"attr6": 6}),
        (2, 3, {"attr7": 7}),
        (3, 4, {"attr8": 8}),
        (4, 5, {"attr9": 9}),
        (1, 5, {"attr10": 10})
    ])

    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 3)])
    graphs = [G1, G2]
    target = np.array([1.0, 2.0])

    return graphs, target


def test_dataset_splitting(nx_graphs, parameters):
    """Check the internals and behaviour of `TemplateEstimator`."""
    path_boost = PathBoost()

    # assert est.demo_param == "demo_param"

    path_boost.fit(nx_graphs[0], nx_graphs[1], list_anchor_nodes_labels=parameters["anchor_nodes"],
                   anchor_nodes_label_name=parameters["anchor_nodes_label_name"])
    assert hasattr(path_boost, "is_fitted_")

    X = nx_graphs[0]
    y_pred = path_boost.predict(X)
    assert_array_equal(y_pred, np.ones(len(X), dtype=np.int64))
