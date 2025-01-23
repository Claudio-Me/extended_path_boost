import networkx as nx
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from extended_path_boost._extended_path_boost import PathBoost
from sklearn.model_selection import train_test_split


def independent_cross_validation_on_each_anchor_node(X: list[nx.Graph], y, param_grid: dict = None):
    # Define the parameter grid
    # TODO remove this hard coded param_grid
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.01, 0.02, 0.05],
            'max_path_length': [3, 5, 7],
            'kwargs_for_base_learner': [{'max_depth': 3}, {'max_depth': 4}]
        }

    # Initialize the PathBoost model
    path_boost = PathBoost(n_iter=10, n_of_cores=10, verbose=False)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=path_boost, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

    # Fit the model on the training data
    grid_search.fit(X_train, y_train, list_anchor_nodes_labels=[25, 47, 48, 80],
                    anchor_nodes_label_name="feature_atomic_number")

    # Print the best parameters and the best score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", -grid_search.best_score_)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print("Test set score: ", test_score)
