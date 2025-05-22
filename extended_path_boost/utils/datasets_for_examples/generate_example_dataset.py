import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from extended_path_boost._extended_path_boost import PathBoost


def generate_n_random_graphs(n, num_nodes, prob, seed):
    graphs = []
    for i in range(n):
        G = nx.erdos_renyi_graph(num_nodes, prob, seed=seed + i)
        graphs.append(G)
    return graphs


def generate_synthetic_graph_dataset(
        n_graphs=100,
        n_nodes=10,
        n_node_features=2,
        n_edge_features=2,  # <-- new argument
        n_paths=3,
        path_length=4,
        noise_std=1.0,
        random_state=42,
        list_anchor_nodes_labels: list | tuple = (0, 1, 2, 3),
        possible_labels: list | tuple = (4, 5, 6, 7, 8, 9)
):
    """
    Generate a synthetic dataset of graphs with node and edge attributes and labels y,
    where y is a weighted sum of node attributes along predetermined paths.

    Returns:
        graphs: list of nx.Graph
        y: np.ndarray
        paths: list of list of node indices (the predetermined paths)
        weights: np.ndarray (weights used for the sum)
    """
    rng = np.random.RandomState(random_state)
    graphs = generate_n_random_graphs(n=n_graphs, num_nodes=n_nodes, prob=0.5, seed=random_state)
    conunt3 = 0
    for G in graphs:
        # Assign random node features
        anchor_node_is_set = False
        for i in G.nodes:
            for f in range(n_node_features):
                if f == 0:
                    if not anchor_node_is_set:
                        random_anchor_label = rng.choice(list_anchor_nodes_labels)
                        G.nodes[i][f'feature_{f}'] = random_anchor_label
                        anchor_node_is_set = True
                    else:
                        G.nodes[i][f'feature_{f}'] = rng.choice(possible_labels)
                else:
                    G.nodes[i][f'feature_{f}'] = rng.uniform(-1, 1)
        # Assign random edge features
        for u, v in G.edges:
            for ef in range(n_edge_features):
                G.edges[u, v][f'edge_feature_{ef}'] = rng.uniform(-1, 1)

    # Predetermined paths (same for all graphs)
    paths = []
    for p in range(n_paths):
        start = rng.randint(0, n_nodes - path_length)
        path = list(range(start, start + path_length))
        paths.append(path)
    # Random weights for each path and each node feature
    weights = rng.uniform(-2, 2, size=(n_paths, n_node_features))
    # Compute y: use all node features along each path
    y = []
    for G in graphs:
        label = 0.0
        for p_idx, path in enumerate(paths):
            for f in range(n_node_features):
                attr_sum = sum(G.nodes[n][f'feature_{f}'] for n in path)
                label += weights[p_idx, f] * attr_sum
        label += rng.normal(0, noise_std)  # Add random noise to the label
        y.append(label)
    y = np.array(y)
    return graphs, y, paths, weights


if __name__ == "__main__":

    list_anchor_nodes_labels = [3]

    possible_labels = [4, 5, 6, 7, 8, 9]

    # Generate synthetic dataset
    nx_graphs, y, paths, weights = generate_synthetic_graph_dataset(
        n_graphs=18,
        n_nodes=10,
        n_node_features=2,
        n_edge_features=2,
        n_paths=3,
        path_length=4,
        noise_std=10.0,
        random_state=42,
        list_anchor_nodes_labels=list_anchor_nodes_labels,
        possible_labels=possible_labels
    )

    parameters_variable_importance: dict = {
        'criterion': 'absolute',  # 'absolute' or 'relative'
        'error_used': 'mse',  # 'mse' or 'mae'
        'use_correlation': False,
        'normalize': True,
    }

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(nx_graphs, y, test_size=0.2, random_state=42)

    # Define evaluation set
    eval_set = [(X_test, y_test)]

    # Fit the model on the training data
    path_boost = PathBoost(
        n_iter=160,
        max_path_length=6,
        learning_rate=0.1,
        n_of_cores=1,
        verbose=True,
        parameters_variable_importance=parameters_variable_importance
    )
    path_boost.fit(
        X=X_train,
        y=y_train,
        eval_set=eval_set,
        list_anchor_nodes_labels=list_anchor_nodes_labels,
        anchor_nodes_label_name="feature_0"
    )

    path_boost.plot_training_and_eval_errors(skip_first_n_iterations=140, plot_eval_sets_error=False)
    path_boost.plot_variable_importance()
