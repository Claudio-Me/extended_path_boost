import pytest
from sklearn.model_selection import train_test_split
from extended_path_boost._extended_path_boost import PathBoost
from extended_path_boost.tests.test_datasets.load_test_dataset import get_nx_test_dataset, get_y

def test_pathboost_with_dataset():
    # Load the dataset
    nx_graphs = get_nx_test_dataset()
    y = get_y()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(nx_graphs, y, test_size=0.2, random_state=42)

    # Initialize the PathBoost model
    model = PathBoost(n_iter=20, max_path_length=3, learning_rate=0.5)

    # Define anchor nodes labels
    list_anchor_nodes_labels = [25, 47, 48, 80]

    # Define evaluation set
    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Fit the model on the training data
    model.fit(X=X_train, y=y_train, eval_set=eval_set, list_anchor_nodes_labels=list_anchor_nodes_labels,
              anchor_nodes_label_name="feature_atomic_number")

    # Check if the model is fitted
    assert model.is_fitted_

    # Check if the training MSE is recorded
    assert len(model.train_mse_) == 20

    # Check if the evaluation MSE is recorded
    assert len(model.mse_eval_set_) == len(eval_set)

    # Check if paths are selected
    assert len(model.models_list_) > 0