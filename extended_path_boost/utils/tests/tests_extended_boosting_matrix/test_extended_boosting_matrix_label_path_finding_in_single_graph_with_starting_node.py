# test_extended_boosting_matrix_label_path_finding_in_single_graph_with_starting_node.py

import networkx as nx
import pytest
from extended_path_boost.utils.classes.extended_boosting_matrix import ExtendedBoostingMatrix


def test_find_labelled_path_in_nx_graph_single_path():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])

    path_labels = [1, 2, 3]
    result = ExtendedBoostingMatrix._find_labelled_path_in_nx_graph_from_starting_node(
        graph, path_labels, id_label_name="label", starting_node=1
    )
    assert set(result) == {(1, 2, 3)}


def test_find_labelled_path_in_nx_graph_multiple_paths():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3}),
        (4, {"label": 3})
    ])
    graph.add_edges_from([(1, 2), (2, 3), (2, 4)])

    path_labels = [1, 2, 3]
    result = ExtendedBoostingMatrix._find_labelled_path_in_nx_graph_from_starting_node(
        graph, path_labels, id_label_name="label", starting_node=1
    )
    assert set(result) == {(1, 2, 3), (1, 2, 4)}


def test_find_labelled_path_in_nx_graph_no_path():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 10})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])

    path_labels = [1, 2, 3]
    result = ExtendedBoostingMatrix._find_labelled_path_in_nx_graph_from_starting_node(
        graph, path_labels, id_label_name="label", starting_node=1
    )
    assert result == []


def test_find_labelled_path_in_nx_graph_starting_node_not_in_path():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph.add_edges_from([(1, 2), (2, 3)])

    path_labels = [1, 2, 3]
    result = ExtendedBoostingMatrix._find_labelled_path_in_nx_graph_from_starting_node(
        graph, path_labels, id_label_name="label", starting_node=2
    )
    assert result == []


def test_find_labelled_path_in_nx_graph_circular_path():
    graph = nx.Graph()
    graph.add_nodes_from([
        (1, {"label": 1}),
        (2, {"label": 2}),
        (3, {"label": 3})
    ])
    graph.add_edges_from([(1, 2), (2, 3), (3, 1)])

    path_labels = [1, 2, 3]
    result = ExtendedBoostingMatrix._find_labelled_path_in_nx_graph_from_starting_node(
        graph, path_labels, id_label_name="label", starting_node=1
    )
    assert set(result) == {(1, 2, 3)}


def test_find_labelled_path_in_nx_graph_disconnected_graph():
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

    path_labels = [1, 2, 3]
    result = ExtendedBoostingMatrix._find_labelled_path_in_nx_graph_from_starting_node(
        graph, path_labels, id_label_name="label", starting_node=4
    )
    assert set(result) == {(4, 5, 6)}


def test_find_labelled_path_in_nx_graph_loop_graph():
    graph = nx.Graph()
    graph.add_nodes_from([
        (2, {"label": 2}),
        (3, {"label": 3}),
        (4, {"label": 3}),
        (5, {"label": 3})
    ])
    graph.add_edges_from([(2, 3), (2, 4), (3, 4), (4, 5), (5, 3)])

    path_labels = [2, 3, 3]
    result = ExtendedBoostingMatrix._find_labelled_path_in_nx_graph_from_starting_node(
        graph, path_labels, id_label_name="label", starting_node=2
    )
    assert set(result) == {(2, 3, 4), (2, 3, 5), (2, 4, 5), (2, 4, 3)}

