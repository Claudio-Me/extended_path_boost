import numpy as np
import pandas as pd
from matplotlib.font_manager import list_fonts

from pandas.core.interchange import dataframe
from typing import Iterable

import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import ast
import numbers
import copy
import ast


class ExtendedBoostingMatrix:
    frequency_column_name: str = "n_times_present"

    def __init__(self):
        pass

    @staticmethod
    def generate_new_columns_from_columns_names(dataset: list[nx.Graph], columns_names: Iterable[str],
                                                main_label_name: str,
                                                ebm_to_be_expanded: pd.DataFrame | None = None,
                                                replace_nan_with=np.nan) -> pd.DataFrame:
        new_columns = None

        # find paths names in column_names
        # for each path find which column in columns_names are related to it
        columns_for_each_path = defaultdict(lambda: [])
        for column in columns_names:
            path_labels = ExtendedBoostingMatrix.get_path_from_column_name(column)
            columns_for_each_path[path_labels].append(column)

        for path_labels, columns_names_referring_to_path in columns_for_each_path.items():
            frequency_column_name = \
                ExtendedBoostingMatrix.generate_name_of_columns_for(path_labels[:-1],
                                                                    [ExtendedBoostingMatrix.frequency_column_name])[0]

            if (ebm_to_be_expanded is not None) and (frequency_column_name in ebm_to_be_expanded.columns):
                frequency_column = ebm_to_be_expanded[frequency_column_name]
            else:
                frequency_column = [1] * len(dataset)

            expanded_columns = ExtendedBoostingMatrix.generate_new_columns_from_path_labels(dataset=dataset,
                                                                                            path_labels=path_labels,
                                                                                            main_label_name=main_label_name,
                                                                                            frequency_column=frequency_column,
                                                                                            replace_nan_with=replace_nan_with)

            filtered_columns_names = [col for col in columns_names_referring_to_path if col in expanded_columns.columns]
            expanded_columns = expanded_columns[filtered_columns_names]

            if new_columns is None:
                new_columns = expanded_columns
            else:
                new_columns = pd.concat([new_columns, expanded_columns], axis=1)

        # Check if new_columns contains all columns_names and add missing columns
        missing_columns = [column for column in columns_names if column not in new_columns.columns]
        missing_df = pd.DataFrame({col: [None] * len(new_columns) for col in missing_columns})
        new_columns = pd.concat([new_columns, missing_df], axis=1)

        return new_columns

    @staticmethod
    def generate_new_columns_from_path_labels(dataset: list[nx.Graph],
                                              path_labels: list, main_label_name: str,
                                              frequency_column, replace_nan_with=np.nan) -> pd.DataFrame:
        # given a list that reppresent the labels of a path it returns the new columns for this path
        if frequency_column is None:
            frequency_column = [1] * len(dataset)

        columns_for_dataframe = defaultdict(lambda: [[] for _ in range(len(dataset))])

        for graph_number, graph in enumerate(dataset):
            if frequency_column[graph_number] > 0:
                paths_found = ExtendedBoostingMatrix.find_labelled_path_in_nx_graph(graph=graph,
                                                                                    path_labels=path_labels,
                                                                                    main_label_name=main_label_name)
                paths_cumulative_attributes = defaultdict(lambda: [])
                for path in paths_found:
                    attributes = ExtendedBoostingMatrix.get_attributes_of_last_part_of_the_path(graph=graph, path=path)
                    for key, value in attributes.items():
                        column_name = \
                            ExtendedBoostingMatrix.generate_name_of_columns_for(path_label=path_labels,
                                                                                attributes=[key])[0]
                        paths_cumulative_attributes[column_name].append(value)
                for key, values in paths_cumulative_attributes.items():
                    columns_for_dataframe[key][
                        graph_number] = ExtendedBoostingMatrix.combine_attribute_value_of_multiple_paths_in_the_same_graph(
                        values)
                frequency_column_name = ExtendedBoostingMatrix.generate_name_of_columns_for(path_label=path_labels,
                                                                                            attributes=[
                                                                                                ExtendedBoostingMatrix.frequency_column_name])[
                    0]
                columns_for_dataframe[frequency_column_name][graph_number] = len(paths_found)

        new_df_columns = pd.DataFrame(columns_for_dataframe)
        new_df_columns = ExtendedBoostingMatrix._remove_empty_list_values_from_df(new_df_columns,
                                                                                  default_value=replace_nan_with)

        return new_df_columns

    @staticmethod
    def new_columns_to_expand_ebm_dataframe_with_path(dataset: list[nx.Graph], selected_path: list | tuple,
                                                      main_label_name: str,
                                                      df_to_be_expanded: pd.DataFrame,
                                                      replace_nan_with=np.nan) -> pd.DataFrame:
        # this function returns the new columns that should be added to the dataframe to expand it. The new columns come from the paths that expands selected path
        # find frequency column relative to the selected path
        path_frequency_column_name = \
            ExtendedBoostingMatrix.generate_name_of_columns_for(selected_path,
                                                                [ExtendedBoostingMatrix.frequency_column_name])[0]

        path_frequency_column = df_to_be_expanded[path_frequency_column_name]

        effective_paths_in_graphs = ExtendedBoostingMatrix.find_paths_in_dataset(dataset=dataset,
                                                                                 path_labels=selected_path,
                                                                                 id_label_name=main_label_name,
                                                                                 frequency_list=path_frequency_column)

        graphs_new_labels = [[] for _ in range(len(dataset))]
        graphs_new_attributes = [[] for _ in range(len(dataset))]
        for graph_number, paths_in_graph in enumerate(effective_paths_in_graphs):

            new_labels = []
            new_attributes = []
            for path in paths_in_graph:
                # find all the possible extensions of the path
                new_paths = ExtendedBoostingMatrix.extend_path(nodes_path=path, graph=dataset[graph_number])
                if new_paths is None:
                    continue
                else:
                    for new_path in new_paths:
                        found_attributes = ExtendedBoostingMatrix.get_attributes_of_last_part_of_the_path(
                            graph=dataset[graph_number],
                            path=new_path)
                        new_attributes.append(found_attributes)
                        new_labels.append(dataset[graph_number].nodes[new_path[-1]][main_label_name])

            graphs_new_labels[graph_number] = new_labels
            graphs_new_attributes[graph_number] = new_attributes

        columns_for_dataframe = defaultdict(lambda: [[] for _ in range(len(dataset))])

        for graph_number in range(len(dataset)):
            new_attributes_from_graph = defaultdict(lambda: [])
            for i, new_attributes_from_one_path in enumerate(graphs_new_attributes[graph_number]):
                label = new_attributes_from_one_path[main_label_name]

                for key, value in new_attributes_from_one_path.items():
                    columns_name = ExtendedBoostingMatrix.generate_name_of_columns_for(selected_path + (label,), [key])[
                        0]
                    new_attributes_from_graph[columns_name].append(value)

                frequency_column_name = \
                    ExtendedBoostingMatrix.generate_name_of_columns_for(selected_path + (label,),
                                                                        [ExtendedBoostingMatrix.frequency_column_name])[
                        0]
                new_attributes_from_graph[frequency_column_name].append(1)

            for key, values in new_attributes_from_graph.items():
                if ExtendedBoostingMatrix.frequency_column_name in key:
                    value = len(values)
                else:
                    value = ExtendedBoostingMatrix.combine_attribute_value_of_multiple_paths_in_the_same_graph(values)
                columns_for_dataframe[key][graph_number] = value

        new_df_columns = pd.DataFrame(columns_for_dataframe)
        new_df_columns = ExtendedBoostingMatrix._remove_empty_list_values_from_df(new_df_columns,
                                                                                  default_value=replace_nan_with)
        return new_df_columns

    @staticmethod
    def extend_path(nodes_path: tuple, graph: nx.Graph):
        """
        Given a list of nodes, this function extends the path by adding the next node that is connected to the last node
        in the path.
        """
        if len(nodes_path) == 0:
            return None
        last_node = nodes_path[-1]
        neighbors = list(graph.neighbors(last_node))
        if len(neighbors) == 0:
            return None
        new_paths = []
        for neighbor in neighbors:
            if neighbor not in nodes_path:
                new_paths.append(nodes_path + (neighbor,))
        if len(new_paths) == 0:
            return None
        return new_paths

    @staticmethod
    def find_paths_in_dataset(dataset: list[nx.Graph], path_labels: list | tuple,
                              id_label_name: str, frequency_list=None) -> list[list]:

        if frequency_list is None:
            frequency_list = [1] * len(dataset)
        assert len(frequency_list) == len(dataset)
        paths_in_graphs = [[] for _ in range(len(dataset))]
        for i, graph in enumerate(dataset):
            if frequency_list[i] > 0:
                paths_in_graphs[i].extend(
                    ExtendedBoostingMatrix.find_labelled_path_in_nx_graph(graph=graph, path_labels=path_labels,
                                                                          main_label_name=id_label_name))

        return paths_in_graphs

    @staticmethod
    def find_labelled_path_in_nx_graph(graph: nx.Graph, path_labels: list, main_label_name: str) -> list[tuple[int]]:
        # find starting nodes
        starting_nodes = []
        for node, attributes in graph.nodes(data=True):
            if attributes.get(main_label_name) == path_labels[0]:
                starting_nodes.append(node)

        found_paths = []
        for starting_node in starting_nodes:
            found_paths.extend(
                ExtendedBoostingMatrix._find_labelled_path_in_nx_graph_from_starting_node(graph=graph,
                                                                                          path_labels=path_labels,
                                                                                          main_label_name=main_label_name,
                                                                                          starting_node=starting_node))

        return found_paths

    @staticmethod
    def _find_labelled_path_in_nx_graph_from_starting_node(graph: nx.Graph, path_labels: list, main_label_name: str,
                                                           starting_node: int,
                                                           path=None, visited_nodes: set | None = None) -> list[
        tuple[int]]:
        paths_found: list = []
        if path is None:
            path = []
        if visited_nodes is None:
            visited_nodes = set()

        if starting_node not in visited_nodes:
            label_of_node = graph.nodes[starting_node].get(main_label_name)
            if label_of_node == path_labels[0]:
                path = path + [starting_node]
            else:
                return []
            visited_nodes.add(starting_node)
        else:
            return []
        if len(path_labels) == 1:
            return [tuple(path)]

        # the next label we are looking for is always in the second position of the array "path_labels" since the first element is the element we just found
        neighbors_with_right_label = [
            neighbor for neighbor in graph.neighbors(starting_node)
            if graph.nodes[neighbor].get(main_label_name) == path_labels[1]
        ]
        for neighbour in neighbors_with_right_label:
            if neighbour not in visited_nodes:
                new_paths = ExtendedBoostingMatrix._find_labelled_path_in_nx_graph_from_starting_node(graph=graph,
                                                                                                      path_labels=path_labels[
                                                                                                                  1:],
                                                                                                      main_label_name=main_label_name,
                                                                                                      starting_node=neighbour,
                                                                                                      path=path,
                                                                                                      visited_nodes=visited_nodes.copy())

                paths_found.extend(new_paths)
        return paths_found

    @staticmethod
    def get_attributes_of_node(graph: nx.Graph, node_id: int) -> dict | None:
        if graph.nodes.get(node_id) is not None:
            nodes_attributes = graph.nodes.get(node_id)
            numeric_nodes_attributes = {k: v for k, v in nodes_attributes.items() if isinstance(v, numbers.Number)}
            return numeric_nodes_attributes
        else:
            return None

    @staticmethod
    def get_edge_attributes_of_nx_graph(graph: nx.Graph, last_edge: tuple) -> dict | None:
        if isinstance(graph, nx.MultiGraph) and len(last_edge) == 2:
            last_edge = (last_edge[0], last_edge[1], 0)
        if graph.edges.get(last_edge) is not None:
            edge_attributes = graph.edges.get(last_edge)
            numeric_edge_attributes = {k: v for k, v in edge_attributes.items() if isinstance(v, numbers.Number)}
            return numeric_edge_attributes
        else:
            return None

    @staticmethod
    def get_attributes_of_last_part_of_the_path(graph: nx.Graph, path: list | tuple) -> dict:
        """
            Retrieves and combines attributes of the LAST node in a given path, and optionally,
            attributes of the edge connecting the last two nodes in the path. The method processes
            both node and edge attributes in a NetworkX graph and merges them into a single
            dictionary.

            Parameters
                graph: nx.Graph
                    The NetworkX graph from which attributes are retrieved.
                path: list[int] | tuple[int]
                    A sequence of node identifiers representing a path in the graph.

            Returns
                dict
                    A dictionary containing the combined attributes of the last node and, if
                    applicable, the edge connecting the last two nodes in the specified path.
        """
        path_attributes = {}

        node_attributes = ExtendedBoostingMatrix.get_attributes_of_node(graph, path[-1])
        if node_attributes is not None:
            path_attributes.update(node_attributes)
        if len(path) > 1:
            edge_attributes = ExtendedBoostingMatrix.get_edge_attributes_of_nx_graph(graph, (path[-2], path[-1]))
            if edge_attributes is not None:
                path_attributes.update(edge_attributes)
        return path_attributes

    @staticmethod
    def generate_name_of_columns_for(path_label: tuple | list, attributes: Iterable | None) -> list[str] | None:

        if attributes is None:
            return None
        else:
            return [str(path_label) + '_' + str(attribute) for attribute in attributes]

    @staticmethod
    def combine_attribute_value_of_multiple_paths_in_the_same_graph(values: list[numbers.Number]):
        return np.mean(values)

    @staticmethod
    def initialize_boosting_matrix_with_anchor_nodes_attributes(dataset: list[nx.Graph],
                                                                list_anchor_nodes_labels: list,
                                                                id_label_name: str,
                                                                replace_nan_with=np.nan) -> pd.DataFrame:

        extended_boosting_matrix_df = None
        for anchor_node_label in list_anchor_nodes_labels:
            if isinstance(anchor_node_label, str) or not hasattr(anchor_node_label, '__iter__'):
                anchor_node_label_as_tuple = tuple([anchor_node_label])
            else:
                anchor_node_label_as_tuple = anchor_node_label

            columns_for_anchor_node = ExtendedBoostingMatrix.generate_new_columns_from_path_labels(dataset=dataset,
                                                                                                   path_labels=anchor_node_label_as_tuple,
                                                                                                   frequency_column=None,
                                                                                                   main_label_name=id_label_name,
                                                                                                   replace_nan_with=replace_nan_with)
            if extended_boosting_matrix_df is None:
                extended_boosting_matrix_df = columns_for_anchor_node
            else:
                extended_boosting_matrix_df = pd.concat([extended_boosting_matrix_df, columns_for_anchor_node], axis=1)

        extended_boosting_matrix_df = ExtendedBoostingMatrix._remove_empty_list_values_from_df(
            extended_boosting_matrix_df, default_value=replace_nan_with)

        return extended_boosting_matrix_df

    @staticmethod
    def _remove_empty_list_values_from_df(df: pd.DataFrame, default_value=np.nan) -> pd.DataFrame:
        modified_df = df.map(lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x)
        # the entries of the columns "(path)_n_times_present" are nan if the path is not present in the graph, we convert nan to 0
        columns_to_replace = [col for col in modified_df.columns if
                              ExtendedBoostingMatrix.frequency_column_name in col]
        modified_df[columns_to_replace] = modified_df[columns_to_replace].fillna(0).astype(int)
        if default_value is not np.nan:
            modified_df = modified_df.fillna(default_value)
        return modified_df

    @staticmethod
    def get_attribute_name_from_column_name(column_name: str) -> str:
        return column_name.split('_', 1)[1]

    @staticmethod
    def get_path_from_column_name(column_name: str) -> tuple:
        string_path = column_name.split('_', 1)[0]
        path = ast.literal_eval(string_path)
        # this assert can be removed it is used during coding to make sure no error happens here
        assert isinstance(path, tuple)
        return path

    @staticmethod
    def get_frequency_boosting_matrix(train_ebm_dataframe: pd.DataFrame) -> pd.DataFrame:
        selected_columns = [column for column in train_ebm_dataframe.columns if
                            ExtendedBoostingMatrix.frequency_column_name in column]
        return train_ebm_dataframe[selected_columns]

    @staticmethod
    def get_columns_related_to_path(path: tuple, columns_names: list[str]) -> list[str]:
        def _is_subtuple(main_tuple: tuple, sub_tuple: tuple) -> bool:
            return sub_tuple == main_tuple[:len(column_path)]

        columns_to_keep = []
        for column in columns_names:
            column_path = ExtendedBoostingMatrix.get_path_from_column_name(column)
            if _is_subtuple(main_tuple=path, sub_tuple=column_path):
                columns_to_keep.append(column)

        return columns_to_keep
