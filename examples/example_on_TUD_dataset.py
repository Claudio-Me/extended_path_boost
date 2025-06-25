from extended_path_boost._extended_path_boost import PathBoost
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx


def load_tud_dataset(name='PROTEINS'):
    """Load a dataset from TUDataset collection and convert to networkx format."""
    dataset = TUDataset(root='.', name='PROTEINS')
    nx_graphs = [to_networkx(data) for data in dataset]
    return nx_graphs, dataset.data.y


if __name__ == "__main__":
    # Load the dataset
    nx_graphs, y = load_tud_dataset()

    parameters_variable_importance: dict = {
        'criterion': 'absolute', # 'absolute' or 'relative'
        'error_used': 'mse', # 'mse' or 'mae'
        'use_correlation': False,
        'normalize': True,
    }

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(nx_graphs, y, test_size=0.2, random_state=42)

    # Initialize the PathBoost model
    path_boost = PathBoost(n_iter=70, max_path_length=6, learning_rate=0.02, n_of_cores=1, verbose=True,
                           parameters_variable_importance=parameters_variable_importance)

    # Define anchor nodes labels
    list_anchor_nodes_labels = [25, 47, 48, 80]

    # Define evaluation set
    eval_set = [(X_test, y_test)]

    # Fit the model on the training data
    path_boost.fit(X=X_train, y=y_train, eval_set=eval_set, list_anchor_nodes_labels=list_anchor_nodes_labels,
                   anchor_nodes_label_name="feature_atomic_number")

    path_boost.plot_training_and_eval_errors(skip_first_n_iterations=True)

    path_boost.plot_variable_importance()
