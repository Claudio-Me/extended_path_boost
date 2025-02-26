import networkx as nx

from .classes.extended_boosting_matrix import ExtendedBoostingMatrix
from .classes.single_metal_center_path_boost import SingleMetalCenterPathBoost


def split_dataset_by_metal_centers(graphs_list: list[nx.Graph], anchor_nodes_label_name: str,
                                   anchor_nodes: list) -> list[list[int]]:
    """
        Splits a list of graphs into subgroups based on anchor node labels.

        This static method takes a list of graphs, an anchor nodes label name,
        and a list of anchor nodes. It iterates through each graph and identifies
        nodes labeled with anchor node labels. It then organizes the indices of
        graphs where such anchor nodes are found into corresponding subgroups.

        Args:
            graphs_list (list[nx.Graph]): A list of networkx Graph objects to be processed.
            anchor_nodes_label_name (str): The name of the attribute used to identify anchor nodes in the graphs.
            anchor_nodes (list): A list of anchor nodes to be used as a reference for grouping.

        Returns:
            list[list[int]]: A list containing sublists of indices corresponding to the grouping
                             of graphs based on the presence of the anchor nodes.
    """

    indices_list = [[] for _ in range(len(anchor_nodes))]
    for index_in_anchor_nodes, anchor_node_label in enumerate(anchor_nodes):
        for i, graph in enumerate(graphs_list):
            path_found = ExtendedBoostingMatrix.find_labelled_path_in_nx_graph(graph=graph,
                                                                               path_labels=anchor_node_label,
                                                                               main_label_name=anchor_nodes_label_name)
            if len(path_found) > 0:
                indices_list[index_in_anchor_nodes].append(i)
    return indices_list


def train_pattern_boosting(input_from_parallelization: tuple) -> SingleMetalCenterPathBoost | None:
    model: SingleMetalCenterPathBoost = input_from_parallelization[0]
    if model is None:
        return None
    X = input_from_parallelization[1]
    y = input_from_parallelization[2]
    list_anchor_nodes_labels: tuple = input_from_parallelization[3]
    name_of_label_attribute = input_from_parallelization[4]
    model.fit(X=X, y=y, eval_set=None, list_anchor_nodes_labels=[list_anchor_nodes_labels],
              name_of_label_attribute=name_of_label_attribute)
    return model


def parallel_predict(input_from_parallelization: tuple):
    model: SingleMetalCenterPathBoost = input_from_parallelization[0]
    X = input_from_parallelization[1]
    if model is None or len(X) == 0:
        return None

    return model.predict(X)


def parallel_predict_step_by_step(input_from_parallelization: tuple):
    model: SingleMetalCenterPathBoost = input_from_parallelization[0]
    X = input_from_parallelization[1]
    if model is None or len(X) == 0:
        return None

    return model.predict_step_by_step(X)
