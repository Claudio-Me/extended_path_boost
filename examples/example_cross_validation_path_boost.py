import sys
sys.path.append('/mn/sarpanitu/ansatte-u6/claudm/PycharmProjects/extended_path_boost/')

from extended_path_boost._extended_path_boost import PathBoost

import os
import json
import uuid
from sklearn.model_selection import GridSearchCV, train_test_split

from examples_utils import get_full_nx_dataset_with_y
from sklearn.model_selection import train_test_split
from extended_path_boost.tests.test_datasets.load_test_dataset import get_nx_test_dataset, get_y

if __name__ == "__main__":
    # Load the dataset
    #nx_graphs , y = get_full_nx_dataset_with_y()
    # Load the dataset
    nx_graphs = get_nx_test_dataset()
    y = get_y()
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(nx_graphs, y, test_size=0.2, random_state=42)

    # Define the parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.8],
        'max_path_length': [3],
        'kwargs_for_base_learner': [{'max_depth': 3}]
    }

    # Initialize the PathBoost model
    path_boost = PathBoost(n_iter=5, n_of_cores=11, verbose=False)

    score = 'neg_mean_squared_error'
    score = None
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=path_boost, param_grid=param_grid, cv=2, scoring=score,
                               verbose=3)

    list_anchor_nodes_labels = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # first block
                                39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # second block
                                57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,  # lanthanides
                                72, 73, 74, 75, 76, 77, 78, 79, 80,  # third block
                                89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,  # actinides
                                104, 105, 106, 107, 108, 109, 110, 111, 112]

    list_anchor_nodes_labels = [25, 47, 48, 80]

    # Fit the model on the training data
    grid_search.fit(X_train, y_train, list_anchor_nodes_labels=list_anchor_nodes_labels,
                    anchor_nodes_label_name="feature_atomic_number")

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
