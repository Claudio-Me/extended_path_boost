import pytest
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from extended_path_boost import PathBoost

@pytest.fixture
def sample_graph_data():
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
    anchor_labels = [(1,), (2,)]
    return graphs, target, anchor_labels

def test_fit_with_one_eval_set_none(sample_graph_data):
    """Test that fit method works with an evaluation set."""
    X, y, anchor_labels = sample_graph_data
    eval_set = [(X, y), ([X[1]], [y[1]])]
    model = PathBoost()
    model.fit(X, y, "label", anchor_labels, eval_set=eval_set)
    assert model.is_fitted_ is True
    assert hasattr(model, "models_list_")
    assert len(model.models_list_) == len(anchor_labels)

def test_fit_sets_is_fitted(sample_graph_data):
    """Test that fit method sets the is_fitted_ attribute."""
    X, y, anchor_labels = sample_graph_data
    model = PathBoost()
    model.fit(X, y, "label", anchor_labels)
    assert model.is_fitted_ is True



def test_fit_sets_models_list(sample_graph_data):
    """Test that fit method sets the models_list_ attribute."""
    X, y, anchor_labels = sample_graph_data
    model = PathBoost()
    model.fit(X, y, "label", anchor_labels)
    assert hasattr(model, "models_list_")
    assert len(model.models_list_) == len(anchor_labels)

def test_fit_with_eval_set(sample_graph_data):
    """Test that fit method works with an evaluation set."""
    X, y, anchor_labels = sample_graph_data
    eval_set = [(X, y)]
    model = PathBoost()
    model.fit(X, y, "label", anchor_labels, eval_set=eval_set)
    assert model.is_fitted_ is True
    assert hasattr(model, "models_list_")
    assert len(model.models_list_) == len(anchor_labels)

def test_fit_with_empty_train_data(sample_graph_data):
    """Test that fit method handles empty training data for some anchor labels."""
    X, y, anchor_labels = sample_graph_data
    model = PathBoost()
    # Modify the data to create an empty training set for one anchor label
    X[0].nodes[1]["label"] = 99
    model.fit(X, y, "label", anchor_labels)
    assert model.is_fitted_ is True
    assert hasattr(model, "models_list_")
    assert len(model.models_list_) == len(anchor_labels)
    assert model.models_list_[0] is None  # The first model should be None due to empty training data

def test_fit_parallel_execution(sample_graph_data):
    """Test that fit method works with parallel execution."""
    X, y, anchor_labels = sample_graph_data
    model = PathBoost(n_of_cores=2)
    model.fit(X, y, "label", anchor_labels)
    assert model.is_fitted_ is True
    assert hasattr(model, "models_list_")
    assert len(model.models_list_) == len(anchor_labels)