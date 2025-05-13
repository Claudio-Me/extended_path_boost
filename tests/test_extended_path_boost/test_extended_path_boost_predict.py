import pytest
import networkx as nx
import numpy as np
from extended_path_boost import PathBoost

@pytest.fixture
def sample_graph_data():
    """Fixture to create sample graph data and target values for tests."""
    G1 = nx.Graph()
    G1.add_nodes_from([
        (1, {"attr1": 1, "label": 2}),
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
    G2.add_nodes_from([
        (0, {"attr1": 1, "label": 1}),
        (1, {"attr2": 2, "label": 1}),
        (3, {"attr3": 3, "label": 1})
    ])
    G2.add_edges_from([(0, 1), (1, 3)])
    graphs = [G1, G2]
    target = np.array([1.0, 2.0])
    anchor_labels = [(1,), (2,)]
    return graphs, target, anchor_labels



@pytest.fixture
def sample_graphs():
    G1 = nx.Graph()
    G1.add_nodes_from([
        (1, {"attr1": 1, "label": 1}),
        (2, {"attr2": 2, "label": 2}),
        (3, {"attr3": 3, "label": 3}),
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
    G2.add_nodes_from([
        (0, {"attr1": 1, "label": 1}),
        (1, {"attr2": 2, "label": 1}),
        (3, {"attr3": 3, "label": 1})
    ])
    G2.add_edges_from([(0, 1), (1, 3)])

    return [G1, G2]


def test_predict(sample_graph_data):
    """Test the predict method of PathBoost after fitting the model."""
    X, y, anchor_labels = sample_graph_data
    model = PathBoost()
    model.fit(X, y, "label", anchor_labels)
    predictions = model.predict(X)
    predictions_stp_by_step = model.predict_step_by_step(X)

    assert len(predictions_stp_by_step[-1]) == len(X)
    assert np.allclose(predictions_stp_by_step[-1], predictions, atol=1e-8)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, float) for pred in predictions)

def test_predict_with_empty_dataset():
    """Test the predict method with an empty dataset."""
    X = []
    y = np.array([])
    anchor_labels = [(1,), (2,)]
    model = PathBoost()
    model.fit(X, y, "label", anchor_labels)
    predictions = model.predict(X)
    predictions_stp_by_step = model.predict_step_by_step(X)

    assert len(predictions_stp_by_step[-1]) == len(X)
    assert np.allclose(predictions_stp_by_step[-1], predictions, atol=1e-8)
    assert len(predictions) == 0

def test_predict_with_multiple_anchor_labels(sample_graph_data):
    """Test the predict method with multiple anchor labels."""
    X, y, anchor_labels = sample_graph_data
    anchor_labels = [(1,), (2,), (3,)]
    model = PathBoost()
    model.fit(X, y, "label", anchor_labels)
    predictions = model.predict(X)
    predictions_stp_by_step = model.predict_step_by_step(X)

    assert len(predictions_stp_by_step[-1]) == len(X)
    assert np.allclose(predictions_stp_by_step[-1], predictions, atol=1e-8)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, float) for pred in predictions)

def test_predict_with_overlapping_subsets(sample_graph_data):
    """Test the predict method with overlapping subsets."""
    X, y, anchor_labels = sample_graph_data
    anchor_labels = [(1,), (2,), (1, 2)]
    model = PathBoost()
    model.fit(X, y, "label", anchor_labels)
    predictions = model.predict(X)
    predictions_stp_by_step = model.predict_step_by_step(X)

    assert len(predictions_stp_by_step[-1]) == len(X)
    assert np.allclose(predictions_stp_by_step[-1], predictions, atol=1e-8)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, float) for pred in predictions)
    # Ensure predictions are averaged correctly
    for i in range(len(predictions)):
        assert predictions[i] == pytest.approx(np.mean([predictions[i]]))

def test_predict_with_unfitted_model(sample_graph_data):
    """Test the predict method raises an error if the model is not fitted."""
    X, y, anchor_labels = sample_graph_data
    model = PathBoost()
    with pytest.raises(ValueError, match="This PathBoost instance is not fitted yet."):
        model.predict(X)




def test_predict_with_empty_datasets(sample_graphs):
    graphs = sample_graphs
    target = np.array([1.0, 2.0])
    anchor_labels = [(1,), (2,)]
    model = PathBoost(n_iter=10, max_path_length=5, learning_rate=0.1, n_of_cores=2)
    model.fit(graphs, target, "label", anchor_labels)

    G3 = nx.Graph()
    G3.add_nodes_from([
        (1, {"attr1": 1, "label": 1}),
        (2, {"attr2": 2, "label": 2}),
        (3, {"attr3": 3, "label": 3}),
        (4, {"attr4": 4, "label": 3}),
        (5, {"attr5": 5, "label": 4})
    ])
    G3.add_edges_from([
        (1, 2, {"attr6": 6}),
        (2, 3, {"attr7": 7}),
        (3, 4, {"attr8": 8}),
        (4, 5, {"attr9": 9}),
        (1, 5, {"attr10": 10})
    ])

    G4 = nx.Graph()
    G4.add_nodes_from([
        (0, {"attr1": 1, "label": 1}),
        (1, {"attr2": 2, "label": 1}),
        (3, {"attr3": 3, "label": 1})
    ])
    G4.add_edges_from([(0, 1), (1, 3)])

    new_graphs = [G3, G4]
    predictions = model.predict(new_graphs)

    assert len(predictions) == len(new_graphs)
    assert all(isinstance(pred, float) for pred in predictions)


