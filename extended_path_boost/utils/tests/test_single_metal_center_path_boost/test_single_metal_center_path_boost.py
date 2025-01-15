import networkx as nx
import numpy as np
import pytest
from extended_path_boost.utils.classes.single_metal_center_path_boost import SingleMetalCenterPathBoost
from sklearn.tree import DecisionTreeRegressor


@pytest.fixture
def sample_graph_data():
    """Fixture to create sample graph data and target values for tests."""
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2)])
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 3)])
    graphs = [G1, G2]
    target = np.array([1.0, 2.0])
    anchor_labels = [("attribute1", [0, 1]), ("attribute2", [1, 2])]
    return graphs, target, anchor_labels


def test_single_metal_center_path_boost_initialization():
    model = SingleMetalCenterPathBoost(
        n_iter=50,
        max_path_length=5,
        learning_rate=0.05,
        BaseLearner=DecisionTreeRegressor,
        kwargs_for_base_learner={"max_depth": 3},
        Selector=DecisionTreeRegressor,
        kwargs_for_selector={"max_features": "sqrt"}
    )
    assert model.n_iter == 50
    assert model.max_path_length == 5
    assert model.learning_rate == 0.05
    assert model.kwargs_for_base_learner == {"max_depth": 3}
    assert model.kwargs_for_selector == {"max_features": "sqrt"}


def test_single_metal_center_path_boost_predict():
    model = SingleMetalCenterPathBoost()
    X = [nx.complete_graph(5), nx.complete_graph(6)]
    y = np.array([1, 0])
    list_anchor_nodes_labels = [[0], [1]]
    name_of_label_attribute = "label"
    model.fit(X, y, list_anchor_nodes_labels, name_of_label_attribute)
    predictions = model.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == len(X)


def test_single_metal_center_path_boost_evaluate():
    model = SingleMetalCenterPathBoost()
    X = [nx.complete_graph(5), nx.complete_graph(6)]
    y = np.array([1, 0])
    list_anchor_nodes_labels = [[0], [1]]
    name_of_label_attribute = "label"
    model.fit(X, y, list_anchor_nodes_labels, name_of_label_attribute)
    evaluation = model.evaluate(X, y)
    assert isinstance(evaluation, list)
    assert len(evaluation) > 0


