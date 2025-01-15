import networkx as nx
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from extended_path_boost.utils.classes.single_metal_center_path_boost import SingleMetalCenterPathBoost
from extended_path_boost.tests.test_datasets.load_test_dataset import get_nx_test_dataset, get_y


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

@pytest.fixture
def sample_graph_data_for_check_eval_equal_training_ebm():
    """Fixture to create sample graph data and target values for tests."""
    np.random.seed(42)
    random_integer = np.random.randint(1, 10)
    targetG1 = 60
    targetG2 = 90
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


def test_fit_eval_set_equal_to_train_dataset(sample_graph_data_for_check_eval_equal_training_ebm):
    """Test that fit method sets the is_fitted_ attribute."""
    X, y, anchor_labels = sample_graph_data_for_check_eval_equal_training_ebm
    booster = SingleMetalCenterPathBoost(n_iter=200, max_path_length=5, learning_rate=0.5, verbose=True)
    eval_set = [(X, y)]
    booster.fit(X, y, anchor_labels, "label", eval_set=eval_set)

    # check that the ebm are created correctly
    common_columns = booster.train_ebm_dataframe_.columns.intersection(
        booster.eval_set_ebm_df_and_target_[0][0].columns)

    # Check if the values in the common columns are equal
    columns_equal = booster.train_ebm_dataframe_[common_columns].equals(
        booster.eval_set_ebm_df_and_target_[0][0][common_columns])


    assert columns_equal



    ebm_dataframe = booster.generate_ebm_for_dataset(dataset=X)

    # check that the ebm are created correctly by comparing the generated one to the one created during the training phase
    common_columns_generated_train_and_train_ebm = booster.train_ebm_dataframe_.columns.intersection(
        ebm_dataframe.columns)

    # Check if the values in the common columns are equal
    columns_equal_generated_train_and_train_ebm = booster.train_ebm_dataframe_[
        common_columns_generated_train_and_train_ebm].equals(
        ebm_dataframe[common_columns_generated_train_and_train_ebm])

    assert columns_equal_generated_train_and_train_ebm

    # check that the ebm are created correctly during the eval phase
    common_columns_eval_ebm_and_train_ebm = booster.train_ebm_dataframe_.columns.intersection(
        booster.eval_set_ebm_df_and_target_[0][0].columns)

    # Check if the values in the common columns are equal
    columns_equal_eval_train_ebm = booster.train_ebm_dataframe_[common_columns_eval_ebm_and_train_ebm].equals(
        booster.eval_set_ebm_df_and_target_[0][0][common_columns_eval_ebm_and_train_ebm])

    assert columns_equal_eval_train_ebm



    # check that the ebm are created correctly by comparing the generated one to the one created during the eval phase
    common_columns_generated_train_and_eval_ebm = booster.eval_set_ebm_df_and_target_[0][0].columns.intersection(
        ebm_dataframe.columns)

    # Check if the values in the common columns are equal
    columns_equal_generated_train_and_eval_ebm = booster.eval_set_ebm_df_and_target_[0][0][
        common_columns_generated_train_and_eval_ebm].equals(
        ebm_dataframe[common_columns_generated_train_and_eval_ebm])

    assert columns_equal_generated_train_and_eval_ebm




def test_fit_sets_is_fitted(sample_graph_data):
    """Test that fit method sets the is_fitted_ attribute."""
    X, y, anchor_labels = sample_graph_data
    model = SingleMetalCenterPathBoost()
    model.fit(X, y, anchor_labels, "attribute1")
    assert model.is_fitted_ is True


def test_fit_sets_name_of_label_attribute(sample_graph_data):
    """Test that fit method sets name_of_label_attribute_ correctly."""
    X, y, anchor_labels = sample_graph_data
    model = SingleMetalCenterPathBoost()
    name_of_label_attribute = "attribute1"
    model.fit(X, y, anchor_labels, name_of_label_attribute)
    assert model.name_of_label_attribute_ == name_of_label_attribute


def test_fit_returns_self(sample_graph_data):
    """Test that fit method returns self."""
    X, y, anchor_labels = sample_graph_data
    model = SingleMetalCenterPathBoost()
    result = model.fit(X, y, anchor_labels, "attribute1")
    assert result is model


def test_fit_sets_train_mse(sample_graph_data):
    """Test that train_mse_ is set after fitting."""
    X, y, anchor_labels = sample_graph_data
    model = SingleMetalCenterPathBoost()
    model.fit(X, y, anchor_labels, "attribute1")
    assert hasattr(model, "train_mse_")


def test_fit_sets_eval_sets_mse(sample_graph_data):
    """Test that eval_sets_mse_ is set after fitting if eval_set is provided."""
    X, y, anchor_labels = sample_graph_data
    eval_set = [([X[0]], [y[0]])]
    model = SingleMetalCenterPathBoost()
    model.fit(X, y, anchor_labels, "attribute1", eval_set)
    assert hasattr(model, "eval_sets_mse_")


def test_fit_sets_columns_names(sample_graph_data):
    """Test that columns_names_ is set after fitting."""
    X, y, anchor_labels = sample_graph_data
    model = SingleMetalCenterPathBoost()
    model.fit(X, y, anchor_labels, "attribute1")
    assert hasattr(model, "columns_names_")






def test_single_metal_center_path_boost_fit_cop():
    # Load the dataset
    nx_graphs = get_nx_test_dataset()
    y = get_y()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(nx_graphs, y, test_size=0.4, random_state=2)

    # Initialize the booster
    booster = SingleMetalCenterPathBoost(n_iter=20, max_path_length=3, learning_rate=0.5)

    # Define anchor nodes labels
    list_anchor_nodes_labels = [25, 47, 48, 80]

    # Define evaluation set
    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Fit the model on the training data
    booster.fit(X=X_train, y=y_train, eval_set=eval_set, list_anchor_nodes_labels=list_anchor_nodes_labels,
                name_of_label_attribute="feature_atomic_number")

    # Check if the model is fitted
    assert booster.is_fitted_

    # Check if the training MSE is recorded
    assert len(booster.train_mse_) == 20

    # Check if the evaluation MSE is recorded
    assert len(booster.eval_sets_mse_) == 20

    # Check if the columns names are set
    assert booster.columns_names_ is not None

    # Check if paths are selected
    assert len(booster.paths_selected_by_epb_) > 0

    ebm_dataframe = booster.generate_ebm_for_dataset(dataset=X_train)

    # check that the ebm are created correctly by comparing the generated one to the one created during the training phase
    common_columns_generated_train_and_train_ebm = booster.train_ebm_dataframe_.columns.intersection(
        ebm_dataframe.columns)

    # Check if the values in the common columns are equal
    columns_equal_generated_train_and_train_ebm = booster.train_ebm_dataframe_[
        common_columns_generated_train_and_train_ebm].equals(
        ebm_dataframe[common_columns_generated_train_and_train_ebm])

    assert columns_equal_generated_train_and_train_ebm

    # check that the ebm are created correctly during the eval phase
    common_columns_eval_ebm_and_train_ebm = booster.train_ebm_dataframe_.columns.intersection(
        booster.eval_set_ebm_df_and_target_[0][0].columns)

    # Check if the values in the common columns are equal
    columns_equal_eval_train_ebm = booster.train_ebm_dataframe_[common_columns_eval_ebm_and_train_ebm].equals(
        booster.eval_set_ebm_df_and_target_[0][0][common_columns_eval_ebm_and_train_ebm])

    assert columns_equal_eval_train_ebm



    # check that the ebm are created correctly by comparing the generated one to the one created during the eval phase
    common_columns_generated_train_and_eval_ebm = booster.eval_set_ebm_df_and_target_[0][0].columns.intersection(
        ebm_dataframe.columns)

    # Check if the values in the common columns are equal
    columns_equal_generated_train_and_eval_ebm = booster.eval_set_ebm_df_and_target_[0][0][
        common_columns_generated_train_and_eval_ebm].equals(
        ebm_dataframe[common_columns_generated_train_and_eval_ebm])

    assert columns_equal_generated_train_and_eval_ebm
