from extended_path_boost._extended_path_boost import PathBoost
from examples_utils import get_full_nx_dataset_with_y
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load the dataset
    nx_graphs, y = get_full_nx_dataset_with_y()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(nx_graphs, y, test_size=0.2, random_state=42)

    # Initialize the PathBoost model
    path_boost = PathBoost(n_iter=7000, max_path_length=6, learning_rate=0.02, n_of_cores=4, verbose=True)

    # Define anchor nodes labels
    list_anchor_nodes_labels = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # first block
                    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # second block
                    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,  # lanthanides
                    72, 73, 74, 75, 76, 77, 78, 79, 80,  # third block
                    89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,  # actinides
                    104, 105, 106, 107, 108, 109, 110, 111, 112]

    # Define evaluation set
    eval_set = [(X_test, y_test)]

    # Fit the model on the training data
    path_boost.fit(X=X_train, y=y_train, eval_set=eval_set, list_anchor_nodes_labels=list_anchor_nodes_labels,
                   anchor_nodes_label_name="feature_atomic_number")

    path_boost.plot_training_and_eval_errors()
