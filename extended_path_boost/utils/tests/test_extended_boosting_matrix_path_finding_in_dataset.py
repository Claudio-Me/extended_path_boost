# File: extended_path_boost/utils/tests/test_extended_boosting_matrix_path_finding_in_dataset.py

import networkx as nx
import pytest
from extended_path_boost.utils.classes.extended_boosting_matrix import ExtendedBoostingMatrix


def test_find_paths_in_dataset_single_graph_single_path():
    graph = nx.MultiGraph()
    graph.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "C"})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    dataset = [graph]
    labelled_path = ["A", "B", "C"]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[(1, 2, 3)]]


def test_find_paths_in_dataset_single_graph_multiple_paths():
    graph = nx.MultiGraph()
    graph.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "C"}),
        (4, {"label": "A"}),
        (5, {"label": "B"}),
        (6, {"label": "C"})
    ])
    graph.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])
    dataset = [graph]
    labelled_path = ["A", "B", "C"]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[(1, 2, 3), (4, 5, 6)]]


def test_find_paths_in_dataset_multiple_graphs():
    graph1 = nx.MultiGraph()
    graph1.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "C"})
    ])
    graph1.add_edges_from([(1, 2), (2, 3)])
    graph2 = nx.MultiGraph()
    graph2.add_nodes_from([
        (4, {"label": "A"}),
        (5, {"label": "B"}),
        (6, {"label": "C"})
    ])
    graph2.add_edges_from([(4, 5), (5, 6)])
    dataset = [graph1, graph2]
    labelled_path = ["A", "B", "C"]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[(1, 2, 3)], [(4, 5, 6)]]


def test_find_paths_in_dataset_no_paths_found():
    graph = nx.MultiGraph()
    graph.add_nodes_from([
        (1, {"label": "X"}),
        (2, {"label": "Y"}),
        (3, {"label": "Z"})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    dataset = [graph]
    labelled_path = ["A", "B", "C"]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[]]


def test_find_paths_in_dataset_partial_labels_match():
    graph = nx.MultiGraph()
    graph.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "X"})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    dataset = [graph]
    labelled_path = ["A", "B", "C"]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[]]


def test_find_paths_in_dataset_complex_graph():
    graph = nx.MultiGraph()
    graph.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "C"}),
        (4, {"label": "A"}),
        (5, {"label": "B"}),
        (6, {"label": "C"}),
        (7, {"label": "D"})
    ])
    graph.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6), (6, 7)])
    dataset = [graph]
    labelled_path = ["A", "B", "C"]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[(1, 2, 3), (4, 5, 6)]]

def test_find_paths_in_dataset_multiple_graphs_only_one_contains_the_path():
    graph1 = nx.MultiGraph()
    graph1.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "C"})
    ])
    graph1.add_edges_from([(1, 2), (2, 3)])
    graph2 = nx.MultiGraph()
    graph2.add_nodes_from([
        (4, {"label": "B"}),
        (5, {"label": "B"}),
        (6, {"label": "C"})
    ])
    graph2.add_edges_from([(4, 5), (5, 6)])
    dataset = [graph1, graph2]
    labelled_path = ["A", "B", "C"]
    label_name = "label"
    result = ExtendedBoostingMatrix.find_paths_in_dataset(dataset, labelled_path, label_name)
    assert result == [[(1, 2, 3)], []]
