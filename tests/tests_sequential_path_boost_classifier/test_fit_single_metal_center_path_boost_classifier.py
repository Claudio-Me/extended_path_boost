import networkx as nx
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from extended_path_boost.utils.classes.sequential_path_boost_classifier import SequentialPathBoostClassifier
from tests.datasets_used_for_tests.load_test_dataset import get_nx_test_dataset, get_y

@pytest.fixture
def sample_graph_data():
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
    target = np.array([1, 0])
    anchor_labels = [(1,), (2,)]
    return graphs, target, anchor_labels

@pytest.fixture
def sample_graph_data_for_check_eval_equal_training_ebm():
    np.random.seed(42)
    random_integer = np.random.randint(1, 10)
    targetG1 = 1
    targetG2 = 0
    G1 = nx.Graph()
    G1.add_nodes_from([
        (1, {"label": 1, "node_attr": 11, }),
        (2, {"label": 30, "node_attr": 12, })
    ])
    G1.add_edges_from([
        (1, 2, {"edge_attr": targetG1}),
    ])
    G2 = nx.Graph()
    G2.add_nodes_from([
        (1, {"label": 1, "node_attr": 11, }),
        (2, {"label": 100, "node_attr": 12, })
    ])
    G2.add_edges_from([
        (1, 2, {"edge_attr": targetG2}),
    ])
    graphs = [G1,  G2]
    target = np.array([targetG1, targetG2])
    anchor_labels = [(1,)]
    return graphs, target, anchor_labels

def test_classifier_fit_eval_set_equal_to_train_dataset(sample_graph_data_for_check_eval_equal_training_ebm):
    X, y, anchor_labels = sample_graph_data_for_check_eval_equal_training_ebm
    booster = SequentialPathBoostClassifier(n_iter=200, max_path_length=5, learning_rate=0.5, verbose=True)
    eval_set = [(X, y)]
    booster.fit(X, y, anchor_labels, "label", eval_set=eval_set)
    common_columns = booster.train_ebm_dataframe_.columns.intersection(
        booster.eval_set_ebm_df_and_target_[0][0].columns)
    columns_equal = booster.train_ebm_dataframe_[common_columns].equals(
        booster.eval_set_ebm_df_and_target_[0][0][common_columns])
    assert columns_equal
    ebm_dataframe = booster.generate_ebm_for_dataset(dataset=X)
    common_columns_generated_train_and_train_ebm = booster.train_ebm_dataframe_.columns.intersection(
        ebm_dataframe.columns)
    columns_equal_generated_train_and_train_ebm = booster.train_ebm_dataframe_[
        common_columns_generated_train_and_train_ebm].equals(
        ebm_dataframe[common_columns_generated_train_and_train_ebm])
    assert columns_equal_generated_train_and_train_ebm
    common_columns_eval_ebm_and_train_ebm = booster.train_ebm_dataframe_.columns.intersection(
        booster.eval_set_ebm_df_and_target_[0][0].columns)
    columns_equal_eval_train_ebm = booster.train_ebm_dataframe_[common_columns_eval_ebm_and_train_ebm].equals(
        booster.eval_set_ebm_df_and_target_[0][0][common_columns_eval_ebm_and_train_ebm])
    assert columns_equal_eval_train_ebm
    common_columns_generated_train_and_eval_ebm = booster.eval_set_ebm_df_and_target_[0][0].columns.intersection(
        ebm_dataframe.columns)
    columns_equal_generated_train_and_eval_ebm = booster.eval_set_ebm_df_and_target_[0][0][
        common_columns_generated_train_and_eval_ebm].equals(
        ebm_dataframe[common_columns_generated_train_and_eval_ebm])
    assert columns_equal_generated_train_and_eval_ebm

def test_classifier_fit_sets_is_fitted(sample_graph_data):
    X, y, anchor_labels = sample_graph_data
    model = SequentialPathBoostClassifier()
    model.fit(X, y, anchor_labels, "attribute1")
    assert model.is_fitted_ is True

def test_classifier_fit_sets_name_of_label_attribute(sample_graph_data):
    X, y, anchor_labels = sample_graph_data
    model = SequentialPathBoostClassifier()
    name_of_label_attribute = "attribute1"
    model.fit(X, y, anchor_labels, name_of_label_attribute)
    assert model.name_of_label_attribute_ == name_of_label_attribute

def test_classifier_fit_returns_self(sample_graph_data):
    X, y, anchor_labels = sample_graph_data
    model = SequentialPathBoostClassifier()
    result = model.fit(X, y, anchor_labels, "attribute1")
    assert result is model

def test_classifier_fit_sets_train_mse(sample_graph_data):
    X, y, anchor_labels = sample_graph_data
    model = SequentialPathBoostClassifier()
    model.fit(X, y, anchor_labels, "attribute1")
    assert hasattr(model, "train_logloss_")

def test_classifier_fit_sets_eval_sets_mse(sample_graph_data):
    X, y, anchor_labels = sample_graph_data
    eval_set = [([X[0]], [y[0]])]
    model = SequentialPathBoostClassifier()
    model.fit(X, y, anchor_labels, "attribute1", eval_set)
    assert hasattr(model, "eval_sets_logloss_")

def test_classifier_fit_sets_columns_names(sample_graph_data):
    X, y, anchor_labels = sample_graph_data
    model = SequentialPathBoostClassifier()
    model.fit(X, y, anchor_labels, "attribute1")
    assert hasattr(model, "columns_names_")

def test_single_metal_center_path_boost_classifier_fit_cop():
    nx_graphs = get_nx_test_dataset()
    y = get_y()
    X_train, X_test, y_train, y_test = train_test_split(nx_graphs, y, test_size=0.4, random_state=2)
    # Binarize targets using the average value as threshold
    avg = np.mean(y)
    y_train = (np.array(y_train) >= avg).astype(int)
    y_test = (np.array(y_test) >= avg).astype(int)
    booster = SequentialPathBoostClassifier(n_iter=20, max_path_length=3, learning_rate=0.5)
    list_anchor_nodes_labels = [25, 47, 48, 80]
    eval_set = [(X_train, y_train), (X_test, y_test)]
    booster.fit(X=X_train, y=y_train, eval_set=eval_set, list_anchor_nodes_labels=list_anchor_nodes_labels,
                anchor_nodes_label_name="feature_atomic_number")
    assert booster.is_fitted_
    assert len(booster.train_logloss_) == 20
    assert len(booster.eval_sets_logloss_) == 2
    assert len(booster.eval_sets_logloss_[0]) == 20
    assert booster.columns_names_ is not None
    assert len(booster.paths_selected_by_epb_) > 0
    ebm_dataframe = booster.generate_ebm_for_dataset(dataset=X_train)
    common_columns_generated_train_and_train_ebm = booster.train_ebm_dataframe_.columns.intersection(
        ebm_dataframe.columns)
    columns_equal_generated_train_and_train_ebm = booster.train_ebm_dataframe_[
        common_columns_generated_train_and_train_ebm].equals(
        ebm_dataframe[common_columns_generated_train_and_train_ebm])
    assert columns_equal_generated_train_and_train_ebm
    common_columns_eval_ebm_and_train_ebm = booster.train_ebm_dataframe_.columns.intersection(
        booster.eval_set_ebm_df_and_target_[0][0].columns)
    columns_equal_eval_train_ebm = booster.train_ebm_dataframe_[common_columns_eval_ebm_and_train_ebm].equals(
        booster.eval_set_ebm_df_and_target_[0][0][common_columns_eval_ebm_and_train_ebm])
    assert columns_equal_eval_train_ebm
    common_columns_generated_train_and_eval_ebm = booster.eval_set_ebm_df_and_target_[0][0].columns.intersection(
        ebm_dataframe.columns)
    columns_equal_generated_train_and_eval_ebm = booster.eval_set_ebm_df_and_target_[0][0][
        common_columns_generated_train_and_eval_ebm].equals(
        ebm_dataframe[common_columns_generated_train_and_eval_ebm])
    assert columns_equal_generated_train_and_eval_ebm
