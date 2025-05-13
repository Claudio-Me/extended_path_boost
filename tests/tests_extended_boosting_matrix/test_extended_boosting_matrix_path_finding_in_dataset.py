# File: extended_path_boost/utils/tests/test_extended_boosting_matrix_path_finding_in_dataset.py

import networkx as nx
import pytest
from extended_path_boost.utils.classes.extended_boosting_matrix import ExtendedBoostingMatrix


def test_find_paths_in_dataset_single_graph_single_path():
    graph = nx.MultiGraph()
    graph.add_nodes_from([
        ('1', {"label": 1}),
        ('2', {"label": 2}),
        ('3', {"label": 3})
    ])
    graph.add_edges_from([('1', '2'), ('2', '3')])
    dataset = [graph]
    labelled_path = [1, 2, 3]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[('1','2', '3')]]


def test_find_paths_in_dataset_single_graph_multiple_paths():
    graph = nx.MultiGraph()
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
    labelled_path = [1, 2, 3]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[(1, 2, 3), (4, 5, 6)]]


def test_find_paths_in_dataset_multiple_graphs():
    graph1 = nx.MultiGraph()
    graph1.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph1.add_edges_from([(1, 2), (2, 3)])
    graph2 = nx.MultiGraph()
    graph2.add_nodes_from([
        (4, {"label": 1}),
        (5, {"label": 2}),
        (6, {"label": 3})
    ])
    graph2.add_edges_from([(4, 5), (5, 6)])
    dataset = [graph1, graph2]
    labelled_path = [1, 2, 3]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[(1, 2, 3)], [(4, 5, 6)]]


def test_find_paths_in_dataset_no_paths_found():
    graph = nx.MultiGraph()
    graph.add_nodes_from([
        (1, {"label": 10}),
        (2, {"label": 11}),
        (3, {"label": 21})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    dataset = [graph]
    labelled_path = [1, 2, 3]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[]]


def test_find_paths_in_dataset_partial_labels_match():
    graph = nx.MultiGraph()
    graph.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 10})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    dataset = [graph]
    labelled_path = [1, 2, 3]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[]]


def test_find_paths_in_dataset_complex_graph():
    graph = nx.MultiGraph()
    graph.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3}),
        (4, {"label": 1}),
        (5, {"label": 2}),
        (6, {"label": 3}),
        (7, {"label": 4})
    ])
    graph.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6), (6, 7)])
    dataset = [graph]
    labelled_path = [1, 2, 3]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[(1, 2, 3), (4, 5, 6)]]

def test_find_paths_in_dataset_multiple_graphs_only_one_contains_the_path():
    graph1 = nx.MultiGraph()
    graph1.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph1.add_edges_from([(1, 2), (2, 3)])
    graph2 = nx.MultiGraph()
    graph2.add_nodes_from([
        (4, {"label": 2}),
        (5, {"label": 2}),
        (6, {"label": 3})
    ])
    graph2.add_edges_from([(4, 5), (5, 6)])
    dataset = [graph1, graph2]
    labelled_path = [1, 2, 3]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[(1, 2, 3)], []]



def test_find_paths_in_dataset_multiple_graphs_same_path():
    graph1 = nx.Graph()
    graph1.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph1.add_edges_from([(1, 2), (2, 3)])

    graph2 = nx.Graph()
    graph2.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph2.add_edges_from([(1, 2), (2, 3)])

    dataset = [graph1, graph2]
    path_labels = [1, 2, 3]
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, path_labels, id_label_name="label")
    assert result == [[(1, 2, 3)], [(1, 2, 3)]]


def test_find_paths_in_dataset_with_frequency_list():
    graph1 = nx.Graph()
    graph1.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph1.add_edges_from([(1, 2), (2, 3)])

    graph2 = nx.Graph()
    graph2.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph2.add_edges_from([(1, 2), (2, 3)])

    dataset = [graph1, graph2]
    path_labels = [1, 2, 3]
    frequency_list = [1, 0]
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, path_labels, id_label_name="label",
                                                          frequency_list=frequency_list)
    assert result == [[(1, 2, 3)], []]


def test_find_paths_in_dataset_empty_dataset():
    dataset = []
    path_labels = [1, 2, 3]
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, path_labels, id_label_name="label")
    assert result == []


def test_find_paths_in_dataset_no_valid_paths_in_graph():
    graph = nx.Graph()
    graph.add_nodes_from([
        ('1', {"label": 10}),
        ('2', {"label": 11}),
        ('3', {"label": 21})
    ])
    graph.add_edges_from([('1', '2'), ('2', '3')])
    dataset = [graph]
    path_labels = ['1', '2', '3']
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, path_labels, id_label_name="label")
    assert result == [[]]
