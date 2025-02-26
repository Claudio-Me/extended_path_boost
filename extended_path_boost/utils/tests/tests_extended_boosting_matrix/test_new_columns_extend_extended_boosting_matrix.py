import networkx as nx
import pandas as pd
import pytest
from extended_path_boost.utils.classes.extended_boosting_matrix import ExtendedBoostingMatrix
import numpy as np

@pytest.fixture
def sample_graph():
    graph = nx.Graph()
    graph.add_nodes_from([
        (0, {"label": 1, "attr": 11}),
        (1, {"label": 2, "attr": 12}),
        (2, {"label": 3, "attr": 13}),
        (3, {"label": 4, "attr": 14}),
    ])
    graph.add_edges_from([(0, 1), (1, 2), (0, 3)])

    return graph


@pytest.fixture
def sample_dataframe():
    data = {"(1,)_n_times_present": [1, 1], "(2,)_n_times_present": [1, 1], "(1, 2)_n_times_present": [1, 1]}
    return pd.DataFrame(data)


@pytest.fixture
def dataset_and_dataframe(sample_graph, sample_dataframe):
    dataset = [sample_graph, sample_graph]  # Replicated graph for testing
    return dataset, sample_dataframe


def test_new_columns_to_expand_ebm_dataframe_with_path(dataset_and_dataframe):
    dataset, df_to_be_expanded = dataset_and_dataframe
    selected_path = (1, 2)
    id_label_name = "label"

    new_columns = ExtendedBoostingMatrix.new_columns_to_expand_ebm_dataframe_with_path(
        dataset=dataset,
        selected_path=selected_path,
        df_to_be_expanded=df_to_be_expanded,
        main_label_name=id_label_name,
        replace_nan_with= np.nan
    )

    assert isinstance(new_columns, pd.DataFrame)
    assert not new_columns.empty
    assert all(col.startswith("(1") or col.startswith("(2") for col in new_columns.columns)
    assert new_columns['(1, 2, 3)_n_times_present'].equals(pd.Series([1, 1]))


def test_new_columns_to_expand_ebm_dataframe_more_complicate_graph():
    graph_1 = nx.Graph()
    graph_1.add_nodes_from([
        (0, {"label": 1, "attr": 11}),
        (1, {"label": 2, "attr": 12}),
        (2, {"label": 3, "attr": 13}),
        (3, {"label": 2, "attr": 14}),
    ])
    graph_1.add_edges_from([(0, 1), (1, 2), (0, 3)])

    graph_2 = nx.Graph()
    graph_2.add_nodes_from([
        (0, {"label": 1, "attr": 11}),
        (1, {"label": 2, "attr": 12}),
        (2, {"label": 2, "attr": 13}),
        (3, {"label": 2, "attr": 14}),
    ])
    graph_2.add_edges_from([(0, 1), (0, 2), (0, 3)])

    dataset = [graph_1, graph_2]

    df_to_be_expanded = {"(1,)_n_times_present": [1, 1]}

    df_to_be_expanded = pd.DataFrame(df_to_be_expanded)
    selected_path = (1,)
    id_label_name = "label"

    new_columns = ExtendedBoostingMatrix.new_columns_to_expand_ebm_dataframe_with_path(
        dataset=dataset,
        selected_path=selected_path,
        df_to_be_expanded=df_to_be_expanded,
        main_label_name=id_label_name,
        replace_nan_with= np.nan
    )

    assert isinstance(new_columns, pd.DataFrame)
    assert not new_columns.empty
    assert all(col.startswith("(1") for col in new_columns.columns)
    assert new_columns['(1, 2)_n_times_present'].equals(pd.Series([2, 3]))
