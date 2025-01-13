import networkx as nx
import pandas as pd
import pytest
from extended_path_boost.utils.classes.extended_boosting_matrix import ExtendedBoostingMatrix
from extended_path_boost.tests.test_datasets import load_test_dataset


def test_on_molecules_dataset():
    dataset = load_test_dataset.get_nx_test_dataset()
    list_anchor_nodes_labels = [25, -1]
    id_label_name = "feature_atomic_number"

    result = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
        dataset, list_anchor_nodes_labels, id_label_name
    )
    assert set(result.columns) == {'(25,)_feature_atomic_number', '(25,)_feature_hydrogen_count',
                                   '(25,)_feature_lone_pair_energy_min_max_difference',
                                   '(25,)_feature_lone_pair_max_d_occupation', '(25,)_feature_lone_pair_max_energy',
                                   '(25,)_feature_lone_pair_max_occupation', '(25,)_feature_lone_pair_max_p_occupation',
                                   '(25,)_feature_lone_pair_max_s_occupation',
                                   '(25,)_feature_lone_vacancy_energy_min_max_difference',
                                   '(25,)_feature_lone_vacancy_min_d_occupation',
                                   '(25,)_feature_lone_vacancy_min_energy', '(25,)_feature_lone_vacancy_min_occupation',
                                   '(25,)_feature_lone_vacancy_min_p_occupation',
                                   '(25,)_feature_lone_vacancy_min_s_occupation', '(25,)_feature_n_lone_pairs',
                                   '(25,)_feature_n_lone_vacancies', '(25,)_feature_natural_atomic_charge',
                                   '(25,)_feature_natural_electron_configuration_d_occupation',
                                   '(25,)_feature_natural_electron_configuration_p_occupation',
                                   '(25,)_feature_natural_electron_configuration_s_occupation',
                                   '(25,)_feature_natural_electron_population_valence', '(25,)_n_times_present',
                                   '(-1,)_n_times_present'}

    assert result.isin([[]]).any().any() == False


def test_initialize_boosting_matrix_for_training_with_node_and_edge_attributes():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": 1, "attr": 10}),
        (2, {"label": 2, "attr": 20}),
        (3, {"label": 3, "attr": 30})
    ])
    graph.add_edges_from([(1, 2, {"weight": 1.0}), (2, 3, {"weight": 2.0})])
    dataset = [graph]
    list_anchor_nodes_labels = [1]
    id_label_name = "label"

    result = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
        dataset, list_anchor_nodes_labels, id_label_name
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert set(result.columns) == {'(1,)_label', '(1,)_attr', '(1,)_n_times_present'}


def test_initialize_boosting_matrix_for_training_single_graph_single_label():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    dataset = [graph]
    list_anchor_nodes_labels = [1]
    id_label_name = "label"

    result = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
        dataset, list_anchor_nodes_labels, id_label_name
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert set(result.columns)  # Expected column names would go here if mocking attributes


def test_initialize_boosting_matrix_for_training_multiple_graphs():
    graph1 = nx.Graph()
    graph1.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph1.add_edges_from([(1, 2), (2, 3)])

    graph2 = nx.Graph()
    graph2.add_nodes_from([
        (4, {"label": 1}),
        (5, {"label": 2})
    ])
    graph2.add_edges_from([(4, 5)])

    dataset = [graph1, graph2]
    list_anchor_nodes_labels = [1]
    id_label_name = "label"

    result = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
        dataset, list_anchor_nodes_labels, id_label_name
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert set(result.columns)  # Expected column names would go here if mocking attributes


def test_initialize_boosting_matrix_for_training_no_paths_found():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": 4}),
        (2, {"label": 5}),
        (3, {"label": 6})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    dataset = [graph]
    list_anchor_nodes_labels = [1]
    id_label_name = "label"

    result = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
        dataset, list_anchor_nodes_labels, id_label_name
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == 1


def test_initialize_boosting_matrix_for_training_multiple_anchor_labels():
    graph_1 = nx.Graph()
    graph_1.add_nodes_from([
        (1, {"label": 1, "attr": 10}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph_1.add_edges_from([(1, 2), (2, 3)])

    graph_2 = nx.Graph()
    graph_2.add_nodes_from([
        (1, {"label": 2}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph_2.add_edges_from([(1, 2), (2, 3)])
    dataset = [graph_1, graph_2]
    list_anchor_nodes_labels = [1, 2]

    id_label_name = "label"

    result = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
        dataset, list_anchor_nodes_labels, id_label_name
    )
    assert set(result.columns) == {'(1,)_label', '(1,)_attr', '(1,)_n_times_present', '(2,)_label',
                                   '(2,)_n_times_present'}
    assert result['(2,)_n_times_present'].equals( pd.Series([1, 2]))
    assert result['(1,)_n_times_present'].equals(pd.Series([1, 0]))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_initialize_boosting_matrix_for_training_empty_dataset():
    dataset = []
    list_anchor_nodes_labels = [1]
    id_label_name = "label"

    result = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
        dataset, list_anchor_nodes_labels, id_label_name
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_initialize_boosting_matrix_for_training_single_graph_multiple_labels():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3}),
        (4, {"label": 1}),
        (5, {"label": 2}),
        (6, {"label": 3})
    ])
    graph.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])
    dataset = [graph]
    list_anchor_nodes_labels = [1, 2]
    id_label_name = "label"

    result = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
        dataset, list_anchor_nodes_labels, id_label_name
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert set(result.columns)  # Expected column names would go here if mocking attributes


def test_initialize_boosting_matrix_for_training_multiple_graphs_multiple_labels():
    graph1 = nx.Graph()
    graph1.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph1.add_edges_from([(1, 2), (2, 3)])

    graph2 = nx.Graph()
    graph2.add_nodes_from([
        (4, {"label": 1}),
        (5, {"label": 2}),
        (6, {"label": 3})
    ])
    graph2.add_edges_from([(4, 5), (5, 6)])

    dataset = [graph1, graph2]
    list_anchor_nodes_labels = [1, 2]
    id_label_name = "label"

    result = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
        dataset, list_anchor_nodes_labels, id_label_name
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert set(result.columns)  # Expected column names would go here if mocking attributes


def test_initialize_boosting_matrix_for_training_no_labels_found():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": 4}),
        (2, {"label": 5}),
        (3, {"label": 6})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    dataset = [graph]
    list_anchor_nodes_labels = [1, 2]
    id_label_name = "label"

    result = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
        dataset, list_anchor_nodes_labels, id_label_name
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == len(list_anchor_nodes_labels)


def test_initialize_boosting_matrix_for_training_with_edge_attributes():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph.add_edges_from([(1, 2, {"weight": 1.0}), (2, 3, {"weight": 2.0})])
    dataset = [graph]
    list_anchor_nodes_labels = [1]
    id_label_name = "label"

    result = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
        dataset, list_anchor_nodes_labels, id_label_name
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert set(result.columns)  # Expected column names would go here if mocking attributes
