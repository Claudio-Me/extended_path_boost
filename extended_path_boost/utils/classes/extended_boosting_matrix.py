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

    def __init__(self, df: pd.DataFrame | None = None):
        self.df: pd.DataFrame | None = df
        if self.df is not None:
            ExtendedBoostingMatrix.sort_df_columns(self.df)

    @staticmethod
    def find_paths_in_dataset(dataset: list[nx.Graph], path_labels: list,
                              id_label_name: str, frequency_list=None) -> list:

        if frequency_list is None:
            frequency_list = [1] * len(dataset)
        assert len(frequency_list) == len(dataset)
        paths_in_graphs = [[] for _ in range(len(dataset))]
        for i, graph in enumerate(dataset):
            if frequency_list[i] > 0:
                paths_in_graphs[i].extend(
                    ExtendedBoostingMatrix.find_labelled_path_in_nx_graph(graph=graph, path_labels=path_labels,
                                                                          id_label_name=id_label_name))

        return paths_in_graphs

    @staticmethod
    def find_labelled_path_in_nx_graph(graph: nx.Graph, path_labels: list, id_label_name: str) -> list[tuple[int]]:
        # find starting nodes
        starting_nodes = []
        for node, attributes in graph.nodes(data=True):
            if attributes.get(id_label_name) == path_labels[0]:
                starting_nodes.append(node)

        found_paths = []
        for starting_node in starting_nodes:
            found_paths.extend(
                ExtendedBoostingMatrix._find_labelled_path_in_nx_graph_from_starting_node(graph=graph,
                                                                                          path_labels=path_labels,
                                                                                          id_label_name=id_label_name,
                                                                                          starting_node=starting_node))

        return found_paths

    @staticmethod
    def _find_labelled_path_in_nx_graph_from_starting_node(graph: nx.Graph, path_labels: list, id_label_name: str,
                                                           starting_node: int,
                                                           path=None, visited_nodes: set | None = None) -> list[
        tuple[int]]:
        paths_found: list = []
        if path is None:
            path = []
        if visited_nodes is None:
            visited_nodes = set()

        if starting_node not in visited_nodes:
            label_of_node = graph.nodes[starting_node].get(id_label_name)
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
            if graph.nodes[neighbor].get(id_label_name) == path_labels[1]
        ]
        for neighbour in neighbors_with_right_label:
            if neighbour not in visited_nodes:
                new_paths = ExtendedBoostingMatrix._find_labelled_path_in_nx_graph_from_starting_node(graph=graph,
                                                                                                      path_labels=path_labels[
                                                                                                                  1:],
                                                                                                      id_label_name=id_label_name,
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
                                                                id_label_name: str, )->pd.DataFrame:

        # this will be a list where the i-th element is a list. This list contains another list, let us call it paths_of_anchor_node.
        # The j-th element of the list paths_of_anchor_node is a list of tuples. Each tuple is a path in the j-th graph that starts from the i-th anchor node
        paths_for_each_anchor_node: list[list[list[tuple]]] = []
        columns_for_dataframe = defaultdict(lambda: [[] for _ in range(len(dataset))])
        for anchor_node_label in list_anchor_nodes_labels:
            if isinstance(anchor_node_label, str) or not hasattr(anchor_node_label, '__iter__'):
                anchor_node_label_as_tuple = tuple([anchor_node_label])
            else:
                anchor_node_label_as_tuple = anchor_node_label

            paths_for_anchor_node = ExtendedBoostingMatrix.find_paths_in_dataset(dataset=dataset,
                                                                                 path_labels=anchor_node_label_as_tuple,
                                                                                 id_label_name=id_label_name)
            assert (all(len(path) == 1 for paths_in_graph in paths_for_anchor_node for path in paths_in_graph))

            paths_for_each_anchor_node.append(paths_for_anchor_node)
            for graph_number, graph in enumerate(dataset):

                cumulative_path_attributes = defaultdict(lambda: [])
                for path in paths_for_anchor_node[graph_number]:
                    specific_path_attributes = ExtendedBoostingMatrix.get_attributes_of_last_part_of_the_path(
                        graph=graph,
                        path=path)
                    for key, value in specific_path_attributes.items():
                        cumulative_path_attributes[key].append(value)

                path_attributes = {
                    key: ExtendedBoostingMatrix.combine_attribute_value_of_multiple_paths_in_the_same_graph(value) for
                    key, value in cumulative_path_attributes.items()}

                # note: here we use the fact that from python 3.7 the keys in dictionaries are ordered

                for attribute_name in path_attributes.keys():
                    column_name = ExtendedBoostingMatrix.generate_name_of_columns_for(
                        path_label=anchor_node_label_as_tuple, attributes=[attribute_name])
                    column_name = column_name[0]
                    columns_for_dataframe[column_name][graph_number] = path_attributes[attribute_name]

                column_name = ExtendedBoostingMatrix.generate_name_of_columns_for(
                    path_label=anchor_node_label_as_tuple, attributes=["n_times_present"])
                column_name = column_name[0]

                columns_for_dataframe[column_name][graph_number] = len(paths_for_anchor_node[graph_number])

        extended_boosting_matrix_df = pd.DataFrame(columns_for_dataframe)
        extended_boosting_matrix_df = extended_boosting_matrix_df.map(lambda x: None if isinstance(x,list) and len(x)==0 else x)
        return extended_boosting_matrix_df

    @staticmethod
    def get_attribute_name_from_column_name(column_name: str) -> str:
        return column_name.split('_', 1)[1]

    @staticmethod
    def get_path_from_column_name(column_name: str) -> tuple:
        string_path = column_name.split('_', 1)[0]
        path = ast.literal_eval(string_path)
        #this assert can be removed it is used during coding to make sure no error happens here
        assert isinstance(path, tuple)
        return path


    @staticmethod
    def get_frequency_boosting_matrix(train_ebm_dataframe: pd.DataFrame)-> pd.DataFrame:
        selected_columns = [column for column in train_ebm_dataframe.columns if 'n_times_present' in column]
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




    # old methods of the class

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    @staticmethod
    def __get_all_possible_node_attributes(dataset: list[nx.classes.multigraph.MultiGraph]) -> set[str]:
        # it returns the name of all the possible attributes on the nodes
        unique_attributes = set()
        for graph in dataset:
            nodes_attributes_list = copy.deepcopy(graph.nodes(data=True))
            for node, attributes in nodes_attributes_list:
                unique_attributes.update(attributes.keys())
        if 'node_position' in unique_attributes:
            unique_attributes.remove('node_position')
        if 'node_label' in unique_attributes:
            unique_attributes.remove('node_label')
        # add "n_times_present", it refers to the attribute "(path)_times_present"
        unique_attributes.add('n_times_present')
        return unique_attributes

    @staticmethod
    def __get_all_possible_edge_attributes(dataset: list[nx.classes.multigraph.MultiGraph]) -> set[str]:
        # it returns the name of all the possible attributes on the edges
        unique_attributes = set()
        for graph in dataset:
            edges_attributes_list = copy.deepcopy(graph.edges(data=True))
            for node_1, node_2, attributes in edges_attributes_list:
                unique_attributes.update(attributes.keys())
        if 'feature_nbo_type' in unique_attributes:
            unique_attributes.remove('feature_nbo_type')
        return unique_attributes

    @staticmethod
    # finds all the possible attributes of the node
    def __get_node_attributes_of_nx_graph(graph, node_id) -> dict | None:
        if str(node_id) in graph.nodes:
            return copy.deepcopy(graph.nodes[str(node_id)])
        else:
            return None

    @staticmethod
    # finds all the possible attributes of the edge
    def __get_edge_attributes_of_nx_graph(graph: nx.Graph, last_edge: tuple) -> dict | None:
        last_edge = (str(last_edge[0]), str(last_edge[1]), 0)
        if last_edge in graph.edges:
            return copy.deepcopy(graph.edges[last_edge])
        else:
            return None

    @staticmethod
    def __get_all_possible_combinations_between_path_and_node_attributes_in_the_dataset(
            dataset: list[nx.classes.multigraph.MultiGraph],
            selected_paths: list[tuple]) -> list[str]:
        # it returns all the possible combination between path and attributes
        unique_attributes = ExtendedBoostingMatrix.__get_all_possible_node_attributes(dataset)

        columns_name = set()
        for path in selected_paths:
            # add the underscore to omogenize the input, otherwise when split('_',1) is called (it is done in another method) it returns only one argument
            for attribute in sorted(unique_attributes):
                columns_name.add(ExtendedBoostingMatrix.__get_column_name(path=path, attribute=attribute))

        return list(columns_name)

    @staticmethod
    def __get_all_possible_combinations_between_path_and_edge_attributes_in_the_dataset(
            dataset: list[nx.classes.multigraph.MultiGraph],
            selected_paths: list[tuple]) -> list[str]:
        # it returns all the possible combination between path and attributes
        unique_attributes = ExtendedBoostingMatrix.__get_all_possible_edge_attributes(dataset)

        columns_name = set()
        for path in selected_paths:
            if len(path) > 1:
                # add the underscore to omogenize the input, otherwise when split('_',1) is called (it is done in another method) it returns only one argument
                for attribute in sorted(unique_attributes):
                    columns_name.add(ExtendedBoostingMatrix.__get_column_name(path=path, attribute=attribute))

        return list(columns_name)

    @staticmethod
    def __get_column_name(path: tuple, attribute: str):
        return str(path) + '_' + attribute

    def plot_sparsity_matrix(self, save_fig=False):
        return self.__plot_sparsity_matrix(self.df, save_fig)

    @staticmethod
    def __plot_sparsity_matrix(df: pd.DataFrame, save_fig=False):
        # Get the sparsity pattern for each SparseDtype column
        locations = []
        for column in df.columns:
            sparse_series = df[column]
            if pd.api.types.is_sparse(sparse_series):
                sparse_array = sparse_series.array
                # Get the indices of the non-zero entries
                non_zero_indices = sparse_array.sp_index.to_int_index().indices
                col_index = df.columns.get_loc(column)
                locations.extend(zip(non_zero_indices, [col_index] * len(non_zero_indices)))

        # Unpack the locations to separate lists of rows and columns
        rows, cols = zip(*locations)

        plt.figure(figsize=(10, 6))
        plt.scatter(cols, rows, alpha=0.5, s=0.01)
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.gca().invert_yaxis()  # Invert the y-axis to match the matrix representation
        if save_fig is True:
            plt.savefig()
        plt.show()

    @staticmethod
    def __column_sort_key(column_name):
        """
        Generate a sorting key for a DataFrame column based on a composite column name.

        The column name is expected to contain a string representation of a tuple of integers,
        followed by an underscore and another arbitrary name. The sorting key is based on the
        length of the tuple and the values within it.

        Parameters:
        column_name (str): The name of the DataFrame column to generate a key for.

        Returns:
        tuple: A tuple where the first element is the length of the tuple and the second element
               is the tuple of integers itself, which will be used for sorting.

        Example:
        >>> ExtendedBoostingMatrix.__column_sort_key("(1,2)_name")
        (2, (1, 2))

        """
        # Split the column name on the underscore to extract the string-tuple and the name
        if column_name == 'target':
            return 0, (0,)
        string_tuple, feature_name = column_name.split('_', 1)

        # Convert the string-tuple to an actual tuple
        tuple_of_ints = ast.literal_eval(string_tuple)
        # Create a sorting key: (length of the tuple, the tuple itself)
        return (len(tuple_of_ints), tuple_of_ints, str(feature_name))

    @staticmethod
    def sort_df_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        sorted_columns = sorted(dataframe.columns, key=ExtendedBoostingMatrix.__column_sort_key)

        dataframe = dataframe.reindex(columns=sorted_columns)
        return dataframe

    @staticmethod
    def __parse_tuple_from_colname(column_name):
        """
        Extract the tuple portion from a DataFrame column name.

        The column name is expected to start with a string representation of a tuple,
        followed by an underscore and then an arbitrary suffix.

        Parameters:
        column_name (str): The column name from which to extract the tuple.

        Returns:
        tuple: The tuple extracted from the column name, as actual integers.

        Example:
        >>> __parse_tuple_from_colname("(1,2,)_some_name")
        (1, 2,)
        """
        if column_name == 'target':
            return (0,)

        string_tuple, _ = column_name.split('_', 1)
        tuple_of_ints = ast.literal_eval(string_tuple)
        return tuple_of_ints

    def associate_paths_to_columns(self, selected_paths):
        """
        Create a dictionary associating tuples in `selected_paths` with DataFrame column indices.

        For each tuple in `selected_paths`, this function finds DataFrame columns whose names
        start with a string representation of a sub-tuple that matches the beginning of the tuple.
        It returns a dictionary where each key is a tuple from `selected_paths`, and the value
        is a list of column indices where a matching sub-tuple is found.

        Parameters:
        df (pd.DataFrame): The DataFrame with columns to be associated with tuples in `selected_paths`.
        selected_paths (list of tuple): A list of tuples for which to find matching DataFrame columns.

        Returns:
        dict: A dictionary mapping each tuple in `selected_paths` to a list of DataFrame column indices.

        Example:
        df_example = pd.DataFrame({
            "(1,2,)_some_name": [1, 2, 3],
            "(1,)_another_name": [4, 5, 6],
            "(2,3,)_different_name": [7, 8, 9],
            "(1,2,3,4)_name": [10, 11, 12]
        })
        selected_paths_example = [(1, 2, 4, 5), (1, 1, 2, 4, 5)]
        >>> associate_paths_to_columns(df_example, selected_paths_example)
        {(1, 2, 4, 5): [0, 3], (1, 1, 2, 4, 5): []}
        """
        tuple_to_column_indices = {}

        # Iterating through selected paths and DataFrame columns to build the dictionary
        for path in selected_paths:
            tuple_to_column_indices[path] = []
            for i, col in enumerate(self.df.columns):
                # Parse the tuple part of the column name
                column_tuple = self.__parse_tuple_from_colname(col)
                # Check if the column tuple is a sub-tuple at the beginning of the current path
                if path[:len(column_tuple)] == column_tuple:
                    tuple_to_column_indices[path].append(i)

        return tuple_to_column_indices

    def get_pandas_dataframe(self) -> pd.DataFrame:
        return self.df

    @staticmethod
    def get_features_interaction_constraints(selected_paths: list[tuple], node_attributes: list[str] | None = None,
                                             edge_attributes: list[str] | None = None,
                                             list_graphs_nx: list[nx.classes.multigraph.MultiGraph] | None = None):
        # it returns a dictionary where to each labelled path is associated a list containing the name of the columns that contains features relative to said path
        if node_attributes is None:
            node_attributes: set[str] = ExtendedBoostingMatrix.__get_all_possible_node_attributes(list_graphs_nx)
        if edge_attributes is None:
            edge_attributes: set[str] = ExtendedBoostingMatrix.__get_all_possible_edge_attributes(list_graphs_nx)

        dict_of_interaction_constraints = dict()
        # first we add the node attributes
        for labelled_path in selected_paths:
            columns_names = []
            for attribute in node_attributes:

                for i in range(len(labelled_path), 0, -1):
                    sub_tuple = labelled_path[:i]
                    columns_names.append(ExtendedBoostingMatrix.__get_column_name(sub_tuple, attribute))
            dict_of_interaction_constraints[labelled_path] = columns_names

        # we add the edge attributes
        for labelled_path in selected_paths:
            if len(labelled_path) > 1:
                columns_names = []
                for attribute in edge_attributes:

                    for i in range(len(labelled_path), 0, -1):
                        sub_tuple = labelled_path[:i]
                        if len(sub_tuple) > 1:
                            columns_names.append(ExtendedBoostingMatrix.__get_column_name(sub_tuple, attribute))
                dict_of_interaction_constraints[labelled_path] += columns_names

        return dict_of_interaction_constraints

    @staticmethod
    def create_extend_boosting_matrix_for(selected_paths: list[tuple],
                                          list_graphs_nx: list[nx.classes.multigraph.MultiGraph],
                                          convert_to_sparse=False) -> pd.DataFrame:
        # we assume the order of observations in boosting matrix is the same as the order in the variable dataset
        assert isinstance(selected_paths, list)

        # function to help the retrival of attributes in nx graphs

        graphsPB_list: list[GraphPB] = [GraphPB.from_GraphNX_to_GraphPB(graph) for graph in list_graphs_nx]

        all_possible_attributes_from_single_graph: set[str] = ExtendedBoostingMatrix.__get_all_possible_node_attributes(
            list_graphs_nx)
        tmp = ExtendedBoostingMatrix.__get_all_possible_edge_attributes(list_graphs_nx)
        all_possible_attributes_from_single_graph.update(
            ExtendedBoostingMatrix.__get_all_possible_edge_attributes(list_graphs_nx))

        # initialize a list of sets that we will use to create the panda's dataframe
        list_rows: list[dict] = []

        for i, graph in enumerate(graphsPB_list):
            # dictionary that contains all the possible values for the same attribute in one graph
            accumulated_attributes = defaultdict(lambda: [])

            for labelled_path in selected_paths:
                numbered_paths_found_in_graph = graph.find_labelled_path(labelled_path=labelled_path)

                for numbered_path in numbered_paths_found_in_graph:
                    attributes: dict = ExtendedBoostingMatrix.__get_node_attributes_of_nx_graph(graph=list_graphs_nx[i],
                                                                                                node_id=numbered_path[
                                                                                                    -1])
                    if len(numbered_path) > 1:
                        edge_attributes = ExtendedBoostingMatrix.__get_edge_attributes_of_nx_graph(
                            graph=list_graphs_nx[i],
                            last_edge=(numbered_path[-2], numbered_path[
                                -1]))
                        if edge_attributes is not None:
                            attributes.update(edge_attributes)

                    # add the column counting the number of time labelled path is present in the graph

                    attributes["n_times_present"] = len(numbered_paths_found_in_graph)

                    for attr in attributes:
                        if attr in all_possible_attributes_from_single_graph:
                            # Accumulate the attribute values in a list
                            accumulated_attributes[
                                ExtendedBoostingMatrix.__get_column_name(labelled_path, attr)].append(
                                attributes[attr])

            # -------------------------------------------------------------------------
            # Calculate the average of all the accumulated values
            complete_attributes = {attr: np.mean(values) if values else None for attr, values in
                                   accumulated_attributes.items()}

            # add response column
            complete_attributes["target"] = list_graphs_nx[i].graph[settings.graph_label_variable]
            # here is its possibleto add graph atributes
            if settings.add_graph_attributes is True:
                graph_attributes = {k: v for k, v in list_graphs_nx[i].graph.items() if
                                    k != settings.graph_label_variable}
                complete_attributes.update(graph_attributes)

            list_rows.append(complete_attributes)
        extended_boosting_matrix_df: pd.DataFrame = pd.DataFrame(list_rows)

        # -------------------------------------------------------------------------------------------------------------
        # some columns might be selected by the previous run in pattern boosting but not found in the new dataset
        # add the columns that are relative to the nodes
        all_possible_columns_name: list[
            str] = ExtendedBoostingMatrix.__get_all_possible_combinations_between_path_and_node_attributes_in_the_dataset(
            list_graphs_nx,
            selected_paths)

        # add the columns that are relative to the edges
        all_possible_columns_name += ExtendedBoostingMatrix.__get_all_possible_combinations_between_path_and_edge_attributes_in_the_dataset(
            list_graphs_nx,
            selected_paths)

        all_possible_columns_name.append("target")
        missed_columns = list(set(all_possible_columns_name) - set(extended_boosting_matrix_df.columns))
        add_dataset = pd.DataFrame(np.nan, index=np.arange(extended_boosting_matrix_df.shape[0]),
                                   columns=missed_columns)
        # extended_boosting_matrix_df[missed_columns] = [np.nan]*len(missed_columns)
        extended_boosting_matrix_df = pd.concat([extended_boosting_matrix_df, add_dataset], axis=1)

        # -------------------------------------------------------------------------------------------------------------

        # make sure that there are no none values in the n_times_present column
        n_times_present_columns = [str(path) + '_' + "n_times_present" for path in selected_paths]
        extended_boosting_matrix_df[n_times_present_columns] = extended_boosting_matrix_df[
            n_times_present_columns].fillna(0)

        # convert into a sparse dataset
        if convert_to_sparse is True:
            extended_boosting_matrix_df = extended_boosting_matrix_df.astype(pd.SparseDtype(float, fill_value=np.nan))
        extended_boosting_matrix_df = ExtendedBoostingMatrix.sort_df_columns(extended_boosting_matrix_df)
        return extended_boosting_matrix_df

    @staticmethod
    def zero_all_elements_except_the_ones_referring_to_path(x_df: pd.DataFrame, y: pd.Series, path: tuple[int],
                                                            dict_of_interaction_constraints: dict) -> (
            pd.DataFrame, pd.Series):
        # it returns a pd.Dataframe that is a deepcopy of the one given in input, but with it puts nan values in the columns that are not referring to the input path
        assert (len(x_df) == len(y))
        assert isinstance(x_df, pd.DataFrame)
        assert isinstance(y, pd.Series)

        columns_to_keep = dict_of_interaction_constraints[path]

        list_of_paths_involved = [ExtendedBoostingMatrix.__parse_tuple_from_colname(column) for column in
                                  columns_to_keep]

        # Find the length of the longest tuples
        max_path_length = max(len(tup) for tup in list_of_paths_involved)

        # Find the indices of all tuples that have the maximum length
        indices_of_longest_tuples = [index for index, tup in enumerate(list_of_paths_involved) if
                                     len(tup) == max_path_length]

        nan_df = pd.DataFrame(np.nan, index=x_df.index, columns=x_df.columns)
        nan_df[columns_to_keep] = x_df[columns_to_keep]
        # nan_df = x_df.mask([column not in columns_to_keep for column in x_df.columns] & (x_df.notnull()), np.nan, inplace=False)

        # remove all the observations that have nan in the

        nan_df = pd.concat([nan_df, y], axis=1)

        # remove the column  n_times_present because it does not have na
        columns_relative_only_to_last_path = [columns_to_keep[index] for index in indices_of_longest_tuples if
                                              columns_to_keep[index].split('_', 1)[1] != 'n_times_present']

        # nan_df.dropna(subset=columns_relative_only_to_last_path, inplace=True)

        zeroed_y = nan_df[y.name]
        zeroed_x_df = nan_df.drop(y.name, axis=1)

        return zeroed_x_df, zeroed_y
