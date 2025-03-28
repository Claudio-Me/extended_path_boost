import networkx as nx
import pytest
from extended_path_boost.utils.classes.extended_boosting_matrix import ExtendedBoostingMatrix
from extended_path_boost.tests.datasets_used_for_tests import load_test_dataset


def test_find_labelled_path_in_molecule_graph():
    dataset = load_test_dataset.get_nx_test_dataset()
    graph = dataset[0]
    #atomic numbers
    path_labels = [25, 16]
    result = ExtendedBoostingMatrix.find_labelled_path_in_nx_graph(graph, path_labels, main_label_name="feature_atomic_number")
    assert result == [('0', '2')]

def test_find_labelled_path_in_nx_graph_single_path():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "C"})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    path_labels = ["A", "B", "C"]
    result = ExtendedBoostingMatrix.find_labelled_path_in_nx_graph(graph, path_labels, main_label_name="label")
    assert result == [(1, 2, 3)]


def test_find_labelled_path_in_nx_graph_no_path():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "D"})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    path_labels = ["A", "B", "C"]
    result = ExtendedBoostingMatrix.find_labelled_path_in_nx_graph(graph, path_labels, main_label_name="label")
    assert result == []


def test_find_labelled_path_in_nx_graph_multiple_paths():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "C"}),
        (4, {"label": "A"}),
        (5, {"label": "B"}),
        (6, {"label": "C"})
    ])
    graph.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])
    path_labels = ["A", "B", "C"]
    result = ExtendedBoostingMatrix.find_labelled_path_in_nx_graph(graph, path_labels, main_label_name="label")
    assert set(result) == {(1, 2, 3), (4, 5, 6)}


def test_find_labelled_path_in_nx_graph_with_disconnected_graph():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "C"}),
        (4, {"label": "A"}),
        (5, {"label": "B"}),
        (6, {"label": "C"})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    path_labels = ["A", "B", "C"]
    result = ExtendedBoostingMatrix.find_labelled_path_in_nx_graph(graph, path_labels, main_label_name="label")
    assert result == [(1, 2, 3)]


def test_find_labelled_path_in_nx_graph_partial_match():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "C"}),
        (4, {"label": "D"})
    ])
    graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
    path_labels = ["B", "C"]
    result = ExtendedBoostingMatrix.find_labelled_path_in_nx_graph(graph, path_labels, main_label_name="label")
    assert result == [(2, 3)]



def test_find_labelled_path_length_1_in_nx_graph():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": "A"}),
        (2, {"label": "B"}),
        (3, {"label": "C"}),
        (4, {"label": "A"}),
        (5, {"label": "B"}),
        (6, {"label": "C"})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])
    path_labels = ["A"]
    result = ExtendedBoostingMatrix.find_labelled_path_in_nx_graph(graph, path_labels, main_label_name="label")
    assert set(result) == {(1,),(4,)}
