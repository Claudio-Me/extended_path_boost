import networkx as nx
import numpy as np
import pytest
from extended_path_boost.tests.test_datasets import load_test_dataset
from extended_path_boost.utils.classes.single_metal_center_path_boost import SingleMetalCenterPathBoost


@pytest.fixture
def sample_graphs_and_labels():
    # Create sample graphs
    G1 = nx.Graph()
    G1.add_node(0, label=1)
    G1.add_node(1, label=0)
    G1.add_edge(0, 1)

    G2 = nx.Graph()
    G2.add_node(0, label=0)
    G2.add_node(1, label=1)
    G2.add_edge(0, 1)

    graphs = [G1, G2]
    labels = np.array([1, 0])

    list_anchor_nodes_labels = [1]
    return graphs, labels, list_anchor_nodes_labels


@pytest.fixture
def smcpb_instance():
    return SingleMetalCenterPathBoost(n_iter=5, max_path_length=3, learning_rate=0.1)


def test_fit_with_valid_inputs(smcpb_instance, sample_graphs_and_labels):
    graphs, labels, list_anchor_nodes_labels = sample_graphs_and_labels
    smcpb_instance.fit(
        X=graphs,
        y=labels,
        list_anchor_nodes_labels=list_anchor_nodes_labels,
        name_of_label_attribute="label"
    )
    assert smcpb_instance.is_fitted_


def test_fit_with_eval_set(smcpb_instance, sample_graphs_and_labels):
    graphs, labels, list_anchor_nodes_labels = sample_graphs_and_labels
    eval_set = [(graphs, labels)]
    smcpb_instance.fit(
        X=graphs,
        y=labels,
        list_anchor_nodes_labels=list_anchor_nodes_labels,
        name_of_label_attribute="label",
        eval_set=eval_set
    )
    assert smcpb_instance.is_fitted_
    assert hasattr(smcpb_instance, "eval_sets_mse_")





def test_fit_with_mismatched_labels_and_graphs(smcpb_instance):
    graphs = [nx.Graph()]
    labels = np.array([1, 0])  # Mismatched length
    list_anchor_nodes_labels = [[]]
    with pytest.raises(ValueError):
        smcpb_instance.fit(
            X=graphs,
            y=labels,
            list_anchor_nodes_labels=list_anchor_nodes_labels,
            name_of_label_attribute="label"
        )


def test_fit_with_real_dataset(smcpb_instance):
    graphs = load_test_dataset.get_nx_test_dataset()

    y = []
    for nx_graph in graphs:
        y.append(nx_graph.graph["target_tzvp_homo_lumo_gap"])

    list_anchor_nodes_labels= [(16,)]

    smcpb_instance.fit(
        X=graphs,
        y=y,
        list_anchor_nodes_labels=list_anchor_nodes_labels,
        name_of_label_attribute="target_tzvp_homo_lumo_gap"
    )
    assert smcpb_instance.is_fitted_
