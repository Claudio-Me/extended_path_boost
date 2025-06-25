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

    # Generate synthetic dataset
    nx_graphs, y, true_paths, true_weights = generate_synthetic_graph_dataset()


    list_anchor_nodes_labels = [0, 1, 2]

    parameters_variable_importance: dict = {
        'criterion': 'absolute',
        'error_used': 'mse',
        'use_correlation': False,
        'normalize': True,
    }

    X_train, X_test, y_train, y_test = train_test_split(nx_graphs, y, test_size=0.25, random_state=42)

    eval_set = [(X_test, y_test)]

    path_boost = PathBoost(
        n_iter=50, # Reduced for quicker example run
        max_path_length=5,
        learning_rate=0.1,
        n_of_cores=1, # Set to >1 for parallel processing if desired
        verbose=True,
        parameters_variable_importance=parameters_variable_importance
    )

    # Fit the model
    # anchor_nodes_label_name must correspond to the feature storing node types ('feature_0')
    path_boost.fit(
        X=X_train,
        y=y_train,
        eval_set=eval_set,
        list_anchor_nodes_labels=list_anchor_nodes_labels,
        anchor_nodes_label_name="feature_0" # Node types are in 'feature_0'
    )
    
    print(f"Generated {len(nx_graphs)} graphs.")
    print(f"Example y values: {y[:5]}")
    print(f"True paths definitions: {true_paths}")
    print(f"True path weights: {true_weights}")

    path_boost.plot_training_and_eval_errors(skip_first_n_iterations=0, plot_eval_sets_error=True)
    if path_boost.parameters_variable_importance is not None and hasattr(path_boost, 'variable_importance_'):
        path_boost.plot_variable_importance(top_n_features=10)
    else:
        print("Variable importance not computed or available.")

    print("Example run finished.")

```

## API Overview

### PathBoost

- `fit(X, y, anchor_nodes_label_name, list_anchor_nodes_labels, eval_set=None)`
- `predict(X)`
- `predict_step_by_step(X)`
- `evaluate(X, y)`
- `plot_training_and_eval_errors(skip_first_n_iterations=True)`
- `plot_variable_importance()`
- **Attributes:**
  - `train_mse_`: Training error (MSE) at each iteration
  - `mse_eval_set_`: Evaluation set error (MSE) at each iteration (if `eval_set` is provided)
  - `variable_importance_`: Variable/path importance scores (if enabled)
  - `is_fitted_`: Whether the model is fitted
  - `models_list_`: List of fitted SequentialPathBoost models (one per anchor node)
  - (Each SequentialPathBoost in `models_list_` exposes the attributes below)

### SequentialPathBoost

- `fit(X, y, list_anchor_nodes_labels, name_of_label_attribute, eval_set=None)`
- `predict(X)`
- `predict_step_by_step(X)`
- `evaluate(X, y)`
- `plot_training_and_eval_errors(skip_first_n_iterations=True)`
- `plot_variable_importance()`
- **Attributes:**
  - `train_mse_`: Training error (MSE) at each iteration
  - `train_mae_`: Training MAE at each iteration
  - `eval_sets_mse_`: Evaluation set error (MSE) at each iteration (if `eval_set` is provided)
  - `eval_sets_mae_`: Evaluation set MAE at each iteration (if `eval_set` is provided)
  - `variable_importance_`: Variable/path importance scores (if enabled)
  - `paths_selected_by_epb_`: Set of selected paths during boosting
  - `columns_names_`: Names of EBM columns/features used
  - `is_fitted_`: Whether the model is fitted

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
