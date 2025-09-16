import networkx as nx
import numpy as np
from extended_path_boost.utils.classes.sequential_path_boost import SequentialPathBoost
import pytest
from tests.datasets_used_for_tests.load_test_dataset import get_nx_test_dataset, get_y





def test_variable_importance_with_loaded_dataset_absolute():
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
    booster.fit(X=X, y=y, list_anchor_nodes_labels=anchor_labels, anchor_nodes_label_name="feature_atomic_number")

    # Verify variable importance
    assert hasattr(booster, 'variable_importance_'), "Variable importance not stored."
    assert len(booster.variable_importance_) > 0, "Variable importance is empty."


def test_variable_importance_with_loaded_dataset_relative():
    # Load dataset
    X = get_nx_test_dataset()
    y = get_y()

    # Initialize with variable importance parameters
    booster = SequentialPathBoost(
        n_iter=10,
        parameters_variable_importance={
            'criterion': 'relative',
            'error_used': 'mse'
        }
    )

    # Fit the model
    anchor_labels = [25, 47, 48, 80]
    booster.fit(X=X, y=y, list_anchor_nodes_labels=anchor_labels, anchor_nodes_label_name="feature_atomic_number")

    # Verify variable importance
    assert hasattr(booster, 'variable_importance_'), "Variable importance not stored."
    assert len(booster.variable_importance_) > 0, "Variable importance is empty."

