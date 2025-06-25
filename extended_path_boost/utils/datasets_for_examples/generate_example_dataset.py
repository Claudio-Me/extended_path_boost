import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split, GridSearchCV
from extended_path_boost._extended_path_boost import PathBoost


def _find_path_instances(graph, path_definition, type_attribute_name='feature_0'):
    """
    Finds all instances of a given path definition in a graph.
    A path definition is a sequence of node types.
    An instance is a sequence of connected node IDs whose types match the definition.
    """
    instances = []
    k = len(path_definition)
    if k == 0:
        return []

    for start_node in graph.nodes:
        if graph.nodes[start_node].get(type_attribute_name) == path_definition[0]:
            # Stack stores (current_node_sequence, current_definition_index)
            dfs_stack = [([start_node], 0)]
            while dfs_stack:
                current_nodes_in_path, def_idx = dfs_stack.pop()

                if def_idx == k - 1:  # Path complete
                    instances.append(list(current_nodes_in_path))
                    continue

                last_node_in_current_path = current_nodes_in_path[-1]
                next_def_idx = def_idx + 1
                expected_next_type = path_definition[next_def_idx]

                for neighbor in graph.neighbors(last_node_in_current_path):
                    if neighbor not in current_nodes_in_path and \
                       graph.nodes[neighbor].get(type_attribute_name) == expected_next_type:

                        new_path_nodes = list(current_nodes_in_path) # Make a copy
                        new_path_nodes.append(neighbor)
                        dfs_stack.append((new_path_nodes, next_def_idx))
    return instances


def generate_synthetic_graph_dataset(
    n_graphs=100,
    avg_n_nodes=15,
    std_n_nodes=5,
    graph_density=0.3,
    n_node_types=5,
    anchor_node_types: list | None = None,  # New parameter
    n_numerical_node_features=2,  # e.g., feature_1, feature_2
    n_edge_features=1,
    n_true_paths=3,
    avg_true_path_length=3,
    std_true_path_length=1,
    numerical_feature_idx_for_label=1, # 0 for feature_0 (type), 1 for feature_1 etc.
    noise_std=0.5,
    random_state=42
):
    """
    Generates a synthetic dataset of graphs.
    Labels 'y' are derived from predefined "true paths" found in the graphs.
    Node types are stored in 'feature_0'. Numerical features are 'feature_1', 'feature_2', ...
    """
    rng = np.random.RandomState(random_state)

    possible_node_types = list(range(n_node_types))

    # Determine which types can start a true path
    valid_starting_types_for_true_paths = possible_node_types
    if anchor_node_types is not None and len(anchor_node_types) > 0:
        # Ensure anchor_node_types are valid w.r.t. n_node_types
        if not all(t in possible_node_types for t in anchor_node_types):
            raise ValueError(f"All anchor_node_types must be within the range [0, {n_node_types-1}]")
        valid_starting_types_for_true_paths = anchor_node_types

    # 1. Define True Paths and their weights
    true_paths_definitions = []
    for _ in range(n_true_paths):
        path_len = int(max(1, rng.normal(avg_true_path_length, std_true_path_length)))
        if path_len == 0:
            continue

        # First element of the path must be from valid_starting_types_for_true_paths
        first_node_type = rng.choice(valid_starting_types_for_true_paths)

        if path_len == 1:
            path_def = tuple([int(first_node_type)])
        else:
            remaining_path_types = rng.choice(possible_node_types, size=path_len - 1)
            path_def = tuple([int(first_node_type)] + [int(x) for x in remaining_path_types])

        # Add all prefixes of the path that start with an anchor type
        for i in range(len(path_def)):
            prefix = path_def[:i + 1]
            if prefix[0] in valid_starting_types_for_true_paths: # Ensure prefix starts with a valid anchor
                 true_paths_definitions.append(prefix)

    true_paths_definitions = list(set(true_paths_definitions)) # Remove duplicates

    # Adjust number of weights if true_paths_definitions changed size due to prefix addition and set conversion
    actual_n_true_paths = len(true_paths_definitions)
    true_path_weights = rng.uniform(-2, 2, size=actual_n_true_paths)

    graphs = []
    y_labels = []

    # Determine the name of the numerical feature to use for label calculation
    # feature_0 is type, feature_1 is the first numerical, etc.
    label_feature_name = f'feature_{numerical_feature_idx_for_label}'

    for i_graph in range(n_graphs):
        # 2. Generate a random graph
        num_nodes = int(max(2, rng.normal(avg_n_nodes, std_n_nodes)))
        G = nx.erdos_renyi_graph(num_nodes, graph_density, seed=random_state + i_graph)
        if not nx.is_connected(G): # Ensure graph is connected for more interesting paths
            G = nx.erdos_renyi_graph(num_nodes, graph_density, seed=random_state + i_graph + n_graphs)
            if not nx.is_connected(G): # If still not connected, take largest component
                 if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                    largest_cc = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest_cc).copy()
                 if G.number_of_nodes() < 2 : # if too small, regenerate a simple one
                     G = nx.path_graph(max(2,num_nodes//2), create_using=nx.Graph())

        # 3. Assign node features
        for node_idx in G.nodes:
            G.nodes[node_idx]['feature_0'] = rng.choice(possible_node_types) # Node type
            for f_idx in range(n_numerical_node_features):
                G.nodes[node_idx][f'feature_{f_idx + 1}'] = rng.uniform(-1, 1)

        # 4. Assign edge features
        for u, v in G.edges:
            for ef_idx in range(n_edge_features):
                G.edges[u, v][f'edge_feature_{ef_idx}'] = rng.uniform(-1, 1)

        # 5. Calculate label y for the graph
        current_y = 0.0
        for path_idx, path_def in enumerate(true_paths_definitions):
            path_weight = true_path_weights[path_idx]
            path_instances = _find_path_instances(G, path_def, type_attribute_name='feature_0')

            for instance_nodes in path_instances:
                feature_sum_on_instance = 0.0
                if label_feature_name == 'feature_0': # if using the type itself
                     feature_sum_on_instance = sum(G.nodes[node][label_feature_name] for node in instance_nodes if label_feature_name in G.nodes[node])
                else: # if using numerical features
                     feature_sum_on_instance = sum(G.nodes[node].get(label_feature_name, 0) for node in instance_nodes)
                current_y += path_weight * feature_sum_on_instance

        current_y += rng.normal(0, noise_std)

        graphs.append(G)
        y_labels.append(current_y)

    return graphs, np.array(y_labels), true_paths_definitions, true_path_weights


if __name__ == "__main__":
    N_NODE_TYPES = 5 # Total distinct types of nodes, e.g., 0, 1, 2, 3, 4
    # Define which node types will be considered as anchors for true path generation AND for PathBoost
    anchor_types_for_generation_and_boosting = [0, 1, 2] # Example: types 0, 1, and 2 are anchors

    # Generate synthetic dataset
    nx_graphs, y, true_paths, true_weights = generate_synthetic_graph_dataset(
        n_graphs=100,
        avg_n_nodes=12,
        std_n_nodes=3,
        graph_density=0.4,
        n_node_types=N_NODE_TYPES,
        anchor_node_types=anchor_types_for_generation_and_boosting, # Pass the anchor types here
        n_numerical_node_features=2, # feature_1, feature_2
        n_edge_features=1,
        n_true_paths=4, # This will be the number of "base" true paths, prefixes will be added
        avg_true_path_length=3,
        std_true_path_length=1,
        numerical_feature_idx_for_label=1, # Use feature_1 for label calculation
        noise_std=0.1,
        random_state=42
    )

    print(f"Generated {len(nx_graphs)} graphs.")
    print(f"Example y values: {y[:5]}")
    print(f"True paths definitions (may include prefixes): {true_paths}")
    print(f"True path weights: {true_weights}")

    list_anchor_nodes_labels = anchor_types_for_generation_and_boosting # Use the same for PathBoost
    anchor_nodes_label_name_for_fitting = "feature_0" # Node types are in 'feature_0'

    parameters_variable_importance: dict = {
        'criterion': 'absolute',
        'error_used': 'mse',
        'use_correlation': False,
        'normalize': True,
    }

    X_train, X_test, y_train, y_test = train_test_split(nx_graphs, y, test_size=0.25, random_state=42)

    eval_set = [(X_test, y_test)]

    # --- GridSearchCV for hyperparameter tuning ---
    print("\nStarting GridSearchCV for hyperparameter tuning...")

    # Define the parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.8],
        'max_path_length': [3, 4],
        'kwargs_for_base_learner': [{'max_depth': 3}, {'max_depth': 5}]
    }

    # Initialize a base PathBoost model for GridSearchCV
    # n_iter is set to a smaller value for faster CV.
    # Other parameters like n_of_cores, verbose are set for CV.
    base_pb_for_cv = PathBoost(
        n_iter=20, # smaller n_iter for quicker CV
        n_of_cores=1,
        verbose=False,
        parameters_variable_importance=None # Disable var importance during CV
    )

    grid_search = GridSearchCV(
        estimator=base_pb_for_cv,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=1
    )

    # Fit GridSearchCV
    # Pass anchor_nodes_label_name and list_anchor_nodes_labels as they are needed by PathBoost.fit
    grid_search.fit(
        X_train, y_train,
        anchor_nodes_label_name=anchor_nodes_label_name_for_fitting,
        list_anchor_nodes_labels=list_anchor_nodes_labels
    )

    print("\nGridSearchCV finished.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score (Negative MSE): {grid_search.best_score_}")

    # --- Fit final model with best parameters ---
    print("\nFitting final PathBoost model with best parameters...")
    best_params_from_cv = grid_search.best_params_

    path_boost_final = PathBoost(
        max_path_length=best_params_from_cv['max_path_length'],
        learning_rate=best_params_from_cv['learning_rate'],
        kwargs_for_base_learner=best_params_from_cv['kwargs_for_base_learner'],
        verbose=True,
        parameters_variable_importance=parameters_variable_importance
    )

    # Fit the final model
    path_boost_final.fit(
        X=X_train,
        y=y_train,
        eval_set=eval_set,
        list_anchor_nodes_labels=list_anchor_nodes_labels,
        anchor_nodes_label_name=anchor_nodes_label_name_for_fitting
    )

    print("\nPlotting results for the final tuned model...")
    path_boost_final.plot_training_and_eval_errors(skip_first_n_iterations=0, plot_eval_sets_error=True)
    if path_boost_final.parameters_variable_importance is not None and hasattr(path_boost_final, 'variable_importance_'):
        path_boost_final.plot_variable_importance(top_n_features=10)
    else:
        print("Variable importance not computed or available for the final model.")

    print("\nExample run with GridSearchCV finished.")
