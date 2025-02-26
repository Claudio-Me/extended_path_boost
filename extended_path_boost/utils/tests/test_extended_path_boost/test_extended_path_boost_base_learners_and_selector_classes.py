import networkx as nx
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from extended_path_boost._extended_path_boost import PathBoost
from extended_path_boost.tests.test_datasets.load_test_dataset import get_nx_test_dataset, get_y

@pytest.fixture
def dataset():
    nx_graphs = get_nx_test_dataset()
    y = get_y()
    return train_test_split(nx_graphs, y, test_size=0.2, random_state=42)





@pytest.fixture
def example_graphs():
    graphs = []
    for i in range(5):
        g = nx.Graph()
        g.add_node(0, label=1)
        g.add_node(1, label=2)
        g.add_edge(0, 1)
        graphs.append(g)
    return graphs

@pytest.fixture
def example_labels():
    return np.array([1, 2, 3, 4, 5])


def test_pathboost_with_linear_model_on_dataset(dataset):
    X_train, X_test, y_train, y_test = dataset
    model = PathBoost(
        n_iter=10,
        max_path_length=3,
        learning_rate=0.1,
        BaseLearnerClass=LinearRegression,
        verbose=False,
        replace_nan_with=-100
    )
    model.fit(X=X_train, y=y_train, anchor_nodes_label_name='feature_atomic_number', list_anchor_nodes_labels=[(25,), (47,)])
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)

def test_pathboost_with_xgboost_model_on_dataset(dataset):
    X_train, X_test, y_train, y_test = dataset
    model = PathBoost(
        n_iter=10,
        max_path_length=3,
        learning_rate=0.1,
        BaseLearnerClass=XGBRegressor,
        SelectorClass=XGBRegressor,
        verbose=False
    )
    model.fit(X=X_train, y=y_train, anchor_nodes_label_name='feature_atomic_number', list_anchor_nodes_labels=[(25,), (47,)])
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)

def test_pathboost_with_linear_model(example_graphs, example_labels):
    model = PathBoost(
        n_iter=10,
        max_path_length=3,
        learning_rate=0.1,
        BaseLearnerClass=LinearRegression,
        verbose=False,
        replace_nan_with=-100
    )
    model.fit(X=example_graphs, y=example_labels, anchor_nodes_label_name='label', list_anchor_nodes_labels=[(1,), (2,)])
    predictions = model.predict(example_graphs)
    assert len(predictions) == len(example_labels)

def test_pathboost_with_xgboost_model(example_graphs, example_labels):
    model = PathBoost(
        n_iter=10,
        max_path_length=3,
        learning_rate=0.1,
        BaseLearnerClass=XGBRegressor,
        SelectorClass=XGBRegressor,
        verbose=False
    )
    model.fit(X=example_graphs, y=example_labels, anchor_nodes_label_name='label', list_anchor_nodes_labels=[(1,), (2,)])
    predictions = model.predict(example_graphs)
    assert len(predictions) == len(example_labels)