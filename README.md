
# Extended Path Boost

Extended Path Boost is a Python library for interpretable machine learning on graph-structured data. It implements the PathBoost and SequentialPathBoost algorithms, which iteratively construct features based on paths in graphs and use boosting to build predictive models. The library is designed for tasks where input data consists of collections of graphs (e.g., molecules, social networks) and supports variable importance analysis for interpretability.

## Features

- **PathBoost**: Ensemble learning over graph paths, partitioned by anchor nodes.
- **SequentialPathBoost**: Boosting with path-based features, iteratively expanding the feature space.
- **Variable Importance**: Quantifies the importance of paths/features in prediction.
- **Parallel Training**: Supports multi-core training for large datasets.
- **Evaluation and Visualization**: Built-in tools for error tracking and variable importance plotting.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/Extended_Path_Boost.git
cd Extended_Path_Boost
pip install -r requirements.txt
```

## Usage Example

Below is a minimal example using the `PathBoost` model:

```python
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from extended_path_boost._extended_path_boost import PathBoost
from extended_path_boost.utils.datasets_for_examples.generate_example_dataset import generate_synthetic_graph_dataset





if __name__ == "__main__":

    list_anchor_nodes_labels = [0, 1, 2, 3]
    possible_labels = [4, 5, 6, 7, 8, 9]

    # Generate synthetic dataset
    nx_graphs, y, paths, weights = generate_synthetic_graph_dataset(
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
        n_iter=100,
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

    path_boost.plot_training_and_eval_errors(skip_first_n_iterations=10)
    path_boost.plot_variable_importance()

```

## API Overview

### PathBoost

- `fit(X, y, anchor_nodes_label_name, list_anchor_nodes_labels, eval_set=None)`
- `predict(X)`
- `predict_step_by_step(X)`
- `evaluate(X, y)`
- `plot_training_and_eval_errors(skip_first_n_iterations=True)`
- `plot_variable_importance()`

### SequentialPathBoost

- `fit(X, y, list_anchor_nodes_labels, name_of_label_attribute, eval_set=None)`
- `predict(X)`
- `predict_step_by_step(X)`
- `evaluate(X, y)`
- `plot_training_and_eval_errors(skip_first_n_iterations=True)`
- `plot_variable_importance()`

## Requirements

- Python 3.10+
- numpy
- pandas
- scikit-learn
- networkx
- matplotlib

(See `requirements.txt` for the full list.)

## Citation

If you use this library in your research, please cite the corresponding paper (add citation here).

## License

BSD 3-Clause License

