from extended_path_boost._extended_path_boost import PathBoost
from extended_path_boost.tests.datasets_used_for_tests.load_test_dataset import get_nx_test_dataset, get_y
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Load the dataset
    nx_graphs = get_nx_test_dataset()
    y = get_y()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(nx_graphs, y, test_size=0.2, random_state=42)

    # Initialize the PathBoost model
    path_boost = PathBoost(n_iter=7000, max_path_length=6, learning_rate=0.02, n_of_cores=1, verbose=True)

    # Define anchor nodes labels
    list_anchor_nodes_labels = [25, 47, 48, 80]

    # Define evaluation set
    eval_set = [(X_test, y_test)]

    # Fit the model on the training data
    path_boost.fit(X=X_train, y=y_train, eval_set=eval_set, list_anchor_nodes_labels=list_anchor_nodes_labels,
                   anchor_nodes_label_name="feature_atomic_number")

    path_boost.plot_training_and_eval_errors()
