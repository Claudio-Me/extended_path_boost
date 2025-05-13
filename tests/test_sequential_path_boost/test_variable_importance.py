import networkx as nx
import numpy as np
from extended_path_boost.utils.classes.sequential_path_boost import SequentialPathBoost
import pytest
from tests.datasets_used_for_tests.load_test_dataset import get_nx_test_dataset, get_y



@pytest.fixture
def data():
    # First graph with 4 nodes
    G1 = nx.Graph()
    G1.add_nodes_from([
        (0, {"label": 10, "feature": 1.0}),
        (1, {"label": 12, "feature": 2.0}),
        (2, {"label": 14, "feature": 3.0}),
        (3, {"label": 16, "feature": 4.0})
    ])
    G1.add_edges_from([
        (0, 1, {"weight": 3.0}),
        (1, 2, {"weight": 2.5}),
        (2, 3, {"weight": 4.0})
    ])

    # Second graph with 3 nodes
    G2 = nx.Graph()
    G2.add_nodes_from([
        (0, {"label": 11, "feature": 2.5}),
        (1, {"label": 13, "feature": 1.5}),
        (2, {"label": 17, "feature": 3.5})
    ])
    G2.add_edges_from([
        (0, 1, {"weight": 2.0}),
        (1, 2, {"weight": 3.0})
    ])

    # Combine graphs into a dataset
    X = [G1, G2]
    y = np.array([15.0, 40.0])
    anchor_labels = [("A",), ("B",)]

    return X, y, anchor_labels


def test_variable_importance_computation(data):

        X = data[0]
        y = data[1]
        anchor_labels = data[2]

        # Initialize the model with variable importance parameters
        booster = SequentialPathBoost(
            n_iter=6,
            parameters_variable_importance={
                'criterion': 'absolute',
                'error_used': 'mse'
            }
        )

        # Fit the model
        booster.fit(X, y, list_anchor_nodes_labels=anchor_labels, name_of_label_attribute='label')

        # Check that variable importance was computed
        assert hasattr(booster, 'variable_importance_'), "Variable importance not stored."
        assert len(booster.variable_importance_)> 0, "Variable importance is empty."




def test_variable_importance_with_loaded_dataset():
    # Load dataset
    X = get_nx_test_dataset()
    y = get_y()

    # Initialize with variable importance parameters
    booster = SequentialPathBoost(
        n_iter=10,
        parameters_variable_importance={
            'criterion': 'absolute',
            'error_used': 'mae'
        }
    )

    # Fit the model
    anchor_labels = [25, 47, 48, 80]
    booster.fit(X=X, y=y, list_anchor_nodes_labels=anchor_labels, name_of_label_attribute="feature_atomic_number")

    # Verify variable importance
    assert hasattr(booster, 'variable_importance_'), "Variable importance not stored."
    assert len(booster.variable_importance_) > 0, "Variable importance is empty."

