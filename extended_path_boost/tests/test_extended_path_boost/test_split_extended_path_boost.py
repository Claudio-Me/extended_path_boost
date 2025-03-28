# File: extended_path_boost/tests/test_split_extended_path_boost.py

import networkx as nx
import pytest
from extended_path_boost._extended_path_boost import PathBoost
from extended_path_boost.utils.wrapper_path_boost_utils import split_dataset_by_metal_centers

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


def test_split_dataset_by_metal_centers_valid_input(example_graphs):
    anchor_labels = [(1,), (2,)]
    result = split_dataset_by_metal_centers(
        graphs_list=example_graphs,
        anchor_nodes_label_name='label',
        anchor_nodes=anchor_labels,
    )
    assert len(result) == len(anchor_labels)
    assert all(isinstance(lst, list) for lst in result)
    assert 0 in result[0]
    assert 0 in result[1]


def test_split_dataset_by_metal_centers_empty_graphs():
    graphs = []
    anchor_labels = [(1,), (2,)]
    result = split_dataset_by_metal_centers(
        graphs_list=graphs,
        anchor_nodes_label_name='label',
        anchor_nodes=anchor_labels,
    )
    assert result == [[] for _ in anchor_labels]


def test_split_dataset_by_metal_centers_no_matches(example_graphs):
    anchor_labels = [(21,), (20,)]
    result = split_dataset_by_metal_centers(
        graphs_list=example_graphs,
        anchor_nodes_label_name='label',
        anchor_nodes=anchor_labels,
    )
    assert result == [[] for _ in anchor_labels]


def test_split_dataset_by_metal_centers_partial_matches(example_graphs):
    additional_graph = nx.Graph()
    additional_graph.add_node(0, label=3)
    example_graphs.append(additional_graph)

    anchor_labels = [(1,), (3,)]
    result = split_dataset_by_metal_centers(
        graphs_list=example_graphs,
        anchor_nodes_label_name='label',
        anchor_nodes=anchor_labels,
    )
    assert len(result) == len(anchor_labels)
    assert all(isinstance(lst, list) for lst in result)
    assert result[1] == [5]


def test_split_dataset_by_metal_centers_different_label_name(example_graphs):
    additional_graph = nx.Graph()
    additional_graph.add_node(0, new_label_name=1)
    example_graphs.append(additional_graph)

    anchor_labels = [(1,), (2,)]
    result = split_dataset_by_metal_centers(
        graphs_list=example_graphs,
        anchor_nodes_label_name='label',
        anchor_nodes=anchor_labels,
    )
    assert len(result) == len(anchor_labels)
    assert all(isinstance(lst, list) for lst in result)
    assert all(5 not in lst for lst in result)