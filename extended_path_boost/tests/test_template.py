"""This file will just show how to write tests for the template classes."""
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose, assert_array_equal
import networkx as nx
import matplotlib.pyplot as plt


from extended_path_boost import  PathBoost

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause


@pytest.fixture
def data():
    graphs = [
        nx.lollipop_graph(4, 6),
        nx.lollipop_graph(3, 20),
        nx.karate_club_graph(),
        nx.turan_graph(6, 2),
        nx.wheel_graph(5)
    ]



    return graphs, np.ones(len(graphs), dtype=np.int64)



@pytest.fixture
def parameters():
    return {"n_iter": 100, "max_path_length": 10, "learning_rate": 0.1, "base_learner": "Tree",
            "selector": "tree", "base_learner_parameters": None}


def test_template_estimator(data, parameters):
    """Check the internals and behaviour of `TemplateEstimator`."""
    est = PathBoost()
    #assert est.demo_param == "demo_param"

    est.fit(data[0], data[1])
    assert hasattr(est, "is_fitted_")

    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred, np.ones(len(X), dtype=np.int64))




