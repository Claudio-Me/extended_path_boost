import sys

sys.path.append('/mn/sarpanitu/ansatte-u6/claudm/PycharmProjects/extended_path_boost/')

from extended_path_boost._extended_path_boost import PathBoost

import os
import json
import uuid
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from extended_path_boost.utils.datasets_for_examples.generate_example_dataset import generate_synthetic_graph_dataset

if __name__ == "__main__":
    N_NODE_TYPES = 5  # Total distinct types of nodes, e.g., 0, 1, 2, 3, 4
    # Define which node types will be considered as anchors for true path generation AND for PathBoost
    anchor_types_for_generation_and_boosting = [0, 1, 2]  # Example: types 0, 1, and 2 are anchors

    # Generate synthetic dataset
    nx_graphs, y, _, _ = generate_synthetic_graph_dataset(
        n_graphs=100,
        avg_n_nodes=12,
        std_n_nodes=3,
        graph_density=0.4,
        n_node_types=N_NODE_TYPES,
        anchor_node_types=anchor_types_for_generation_and_boosting,  # Pass the anchor types here
        n_numerical_node_features=2,  # feature_1, feature_2
        n_edge_features=1,
        n_true_paths=4,  # This will be the number of "base" true paths, prefixes will be added
        avg_true_path_length=3,
        std_true_path_length=1,
        numerical_feature_idx_for_label=1,  # Use feature_1 for label calculation
        noise_std=0.2,
        random_state=42
    )



    X_train, X_test, y_train, y_test = train_test_split(nx_graphs, y, test_size=0.25, random_state=42)

    eval_set = [(X_test, y_test)]


    # Define the parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.8],
        'max_path_length': [3],
        'kwargs_for_base_learner': [{'max_depth': 5}],
        'patience': [5, 10],
    }

    # Initialize the PathBoost model
    path_boost = PathBoost(n_iter=100, n_of_cores=11, verbose=False)

    score = 'neg_mean_squared_error'
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=path_boost, param_grid=param_grid, cv=2, scoring=score,
                               verbose=3)



    # Fit the model on the training data
    grid_search.fit(X_train, y_train, list_anchor_nodes_labels=anchor_types_for_generation_and_boosting,
                    anchor_nodes_label_name="feature_0")

    # Print the best parameters and the best score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", -grid_search.best_score_)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print("Test set score: ", test_score)

    # Save the results to a JSON file
    results = {
        "best_params": grid_search.best_params_,
        "best_score": -grid_search.best_score_,
        "test_score": test_score
    }

    # Create the directory if it doesn't exist
    results_dir = 'examples_data/examples_results'
    os.makedirs(results_dir, exist_ok=True)

    # Generate a unique file name
    file_name = f"cross_validation_results_{uuid.uuid4().hex}.json"
    file_path = os.path.join(results_dir, file_name)

    # Save the results to the JSON file
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {file_path}")
