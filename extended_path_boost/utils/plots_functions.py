from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_training_and_eval_errors(learning_rate: float, train_mse: list,
                                  mse_eval_set: list | None = None, skip_first_n_iterations: int | bool = False,
                                  show=True, save=False):
    """
    Plots the training and evaluation set errors over iterations.
    """
    # skip_the_first n iterations
    if isinstance(skip_first_n_iterations, bool):
        if skip_first_n_iterations:
            n = int(2 / learning_rate)
        else:
            n = 0
    else:
        n = skip_first_n_iterations

    if len(train_mse) > n:
        train_mse = train_mse[n:]
    else:
        train_mse = train_mse

    plt.figure(figsize=(12, 6))

    # Plot training errors
    plt.plot(range(n, len(train_mse) + n), train_mse, label='Training Error', marker='')

    # Plot evaluation set errors if available
    if mse_eval_set is not None:
        if len(mse_eval_set[0]) > n:
            eval_set_mse = [mse_eval_set[i][n:] for i in range(len(mse_eval_set))]
        else:
            eval_set_mse = mse_eval_set

        num_iterations = len(eval_set_mse[0])
        num_eval_sets = len(eval_set_mse)
        for eval_set_index in range(num_eval_sets):
            if eval_set_mse[eval_set_index][0] is not None:
                plt.plot(range(n, num_iterations + n), eval_set_mse[eval_set_index],
                         label=f'Evaluation Set {eval_set_index + 1}', marker='')

    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Evaluation Set Errors Over Iterations')
    plt.legend()
    plt.grid(True)

    # Ensure x-axis only shows integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if show:
        plt.show()

    if save:
        plt.savefig('training_and_eval_errors.png')


def plot_variable_importance_utils(variable_importance: dict, parameters_variable_importance: dict):
    """
    Plots the variable importance scores.
    """

    assert isinstance(variable_importance, dict), "Variable importance should be a dictionary."

    sorted_items = sorted(variable_importance.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_items)

    # Convert tuples in labels to strings
    labels = [",".join(map(str, label)) if isinstance(label, tuple) else str(label) for label in labels]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color='skyblue')
    plt.xlabel('Importance Score')
    plt.title(parameters_variable_importance['criterion'] + ' Variable Importance')
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
