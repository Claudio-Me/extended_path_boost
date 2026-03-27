import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .classes.sequential_path_boost import SequentialPathBoost

import numpy as np
import pandas as pd
from collections import defaultdict
from .classes.extended_boosting_matrix import ExtendedBoostingMatrix
from sklearn.metrics import mean_squared_error, mean_absolute_error


class VariableImportance_ForSequentialPathBoost:
    # this class is used to compute the variable importance according to the path boost algorithm
    # note, that for the relative variable importance, we need hevvy rely on the fact that the loss function passed
    # to the base learner is mse, if changed we can not guarantee the correct behaviour of the algorithm

    def __init__(self, criterion: str, use_correlation: bool = False, normalize: bool = True, error_used: str = 'mse',
                 normalization_value: float = 100):
        assert error_used == 'mse' or error_used == 'mae', f"error must be either mse or mae, but got {error_used}"
        assert criterion == 'absolute' or criterion == 'relative', f"criterion must be either absolute or relative, but got {criterion}"

        self.error_used = error_used
        self.criterion = criterion
        self.use_correlation = use_correlation
        self.normalize = normalize

        self.normalization_value = normalization_value

        criterion_choices = {
            'absolute': self.compute_absolute_variable_importance,
            'relative': self.compute_relative_variable_importance
        }
        self.compute_variable_importance = criterion_choices[criterion]

        # all the selected paths in order of selection
        self.selected_path_at_iteration = []
        self.columns_at_iteration = []
        self.gradient_at_iteration = []

    def _update(self, path_boost: 'SequentialPathBoost', selected_path: tuple, iteration_number: int,
                gradient: np.ndarray | None = None, ):
        # update is used during training in sequential path boost to save, at each iteration the parameters needed later for the computation of the path importance
        # NB we expect that gradient is just y if we are in the first (0-th) iteration
        # NB we expect that absolute_error is None if we are in the first (0-th) iteration

        # check that the base learner has been already trained in this iteration
        assert (iteration_number == len(
            path_boost.base_learner_.train_mse)), f"iteration number {iteration_number} does not match the number of base learners {len(path_boost.base_learner_.train_mse)}"

        # check that the ebm has not been expanded yet
        assert path_boost._ebm_has_been_expanded_in_this_iteration is False

        columns_names = path_boost.train_ebm_dataframe_.columns

        self.selected_path_at_iteration.append(selected_path)
        self.columns_at_iteration.append(columns_names)

        self.gradient_at_iteration.append(gradient)

    def compute_absolute_variable_importance(self, path_boost: 'SequentialPathBoost') -> dict:
        # compute importance by error improvement
        # if we are in the first iteration there is no previous error to compare with, then we set it equal to the second eror improvement
        # note: one can think to compare the first iteration with the error that we would have if we would have used
        # the mean of the labels (that is the variance of the labels in the case of MSE) however this does not work since
        # the bae learner is limited by the learining rate, sometimes making it first training even worse than the mean

        # check that the iteration number is correct
        # if the iteration number does not coincide with the number of base learners (same as the number of errors)
        # it means we are in a new iteration, but we still have to train the base_learner

        error_improvement = defaultdict(float)
        previous_improvement = 0
        for iteration in range(path_boost.n_iter):

            if iteration == 0:
                # in the first iteration we do not have a previous error to compare with so we skip
                pass
            else:
                path = self.selected_path_at_iteration[iteration]
                if self.error_used == 'mse':
                    improvement = path_boost.train_mse_[iteration - 1] - path_boost.train_mse_[iteration]
                    if improvement < 0 and previous_improvement > 0:
                        print(
                            f"error improvement between iteration {iteration} and {iteration - 1} is negative ({improvement}). This is expected by the algorithm, but it might be a sign of overfitting even if we are comparing the improvement on the train error")
                    error_improvement[path] += improvement

                elif self.error_used == 'mae':
                    improvement = path_boost.train_mae_[iteration - 1] - path_boost.train_mae_[iteration]
                    if improvement < 0 and previous_improvement > 0:
                        print(
                            f"error improvement between iteration {iteration} and {iteration - 1} is negative. This is expected in by the algorithm, but it might be a sign of overfitting even tho we are comparing the improvement on the train error")
                    error_improvement[path] += improvement
                previous_improvement = improvement

                if iteration == 1:
                    # since we did not set any importance for the path selected in the zeroth iteration,
                    # we now set it equal to the importance assignet to the second-selected path
                    first_selected_path = self.selected_path_at_iteration[0]
                    if self.error_used == 'mse':
                        error_improvement[first_selected_path] = path_boost.train_mse_[0] - \
                                                                 path_boost.train_mse_[1]
                    elif self.error_used == 'mae':
                        error_improvement[first_selected_path] = path_boost.train_mae_[0] - \
                                                                 path_boost.train_mae_[1]

        dict_error_improvement = self._get_correlation_and_normalize_if_needed(path_boost=path_boost,
                                                                               error_improvement=error_improvement)

        return dict_error_improvement

    def compute_relative_variable_importance(self, path_boost: 'SequentialPathBoost') -> dict:
        # this is a relative measure of importance
        # it is computed as the ratio between the error improvement of a the second best path and the error improvement of the best path

        error_improvement = defaultdict(float)
        for iteration in range(path_boost.n_iter):
            selected_path_at_iteration = self.selected_path_at_iteration[iteration]

            train_ebm_dataframe_at_iteration = path_boost.train_ebm_dataframe_[self.columns_at_iteration[iteration]]

            frequency_matrix_at_iteration = ExtendedBoostingMatrix.get_frequency_boosting_matrix(
                train_ebm_dataframe_at_iteration)

            frequency_path_name = ExtendedBoostingMatrix.generate_frequency_column_name_for_path(
                path_label=selected_path_at_iteration)
            frequency_matrix_without_best_path = frequency_matrix_at_iteration.drop(frequency_path_name,
                                                                                    axis=1, inplace=False)

            gradient = self.gradient_at_iteration[iteration]

            # get the second-best path
            if iteration == 0:
                # in the first iteration we do not have a previous error to compare with so we skip
                continue

            second_best_path = path_boost._find_best_path(train_ebm_dataframe=frequency_matrix_without_best_path,
                                                          y=gradient,
                                                          SelectorClass=path_boost.SelectorClass,
                                                          kwargs_for_selector=path_boost.kwargs_for_selector)

            # fit a new base learner on the second-best path
            columns_to_keep = ExtendedBoostingMatrix.get_columns_related_to_path(second_best_path,
                                                                                 train_ebm_dataframe_at_iteration.columns)
            restricted_df = train_ebm_dataframe_at_iteration[columns_to_keep]

            new_base_learner = path_boost.BaseLearnerClass(**path_boost.kwargs_for_base_learner)

            new_base_learner.fit(restricted_df, gradient)
            if self.error_used == 'mse':
                new_base_learner_prediction = path_boost.learning_rate * pd.Series(
                    new_base_learner.predict(restricted_df))
                new_base_learner_error = mean_squared_error(y_true=gradient,
                                                            y_pred=new_base_learner_prediction)

                error_difference = new_base_learner_error - path_boost.train_mse_[iteration]

            elif self.error_used == 'mae':
                new_base_learner_prediction = path_boost.learning_rate * pd.Series(
                    new_base_learner.predict(restricted_df))
                new_base_learner_error = mean_absolute_error(y_true=gradient,
                                                             y_pred=new_base_learner_prediction)

                error_difference = new_base_learner_error - path_boost.train_mae_[iteration]

            # update the error improvement
            error_improvement[selected_path_at_iteration] += error_difference

        dict_error_improvement = self._get_correlation_and_normalize_if_needed(path_boost=path_boost,
                                                                               error_improvement=error_improvement)

        return dict_error_improvement

    def _get_correlation_and_normalize_if_needed(self, path_boost: 'SequentialPathBoost',
                                                 error_improvement: dict) -> dict:
        dict_error_improvement = dict(error_improvement)

        if self.use_correlation:
            # we need to compute the correlation between the paths
            dict_error_improvement = self.correlation_importance(path_boost, dict_error_improvement)

        if self.normalize:
            total_error_improvement = sum(dict_error_improvement.values())
            for path in dict_error_improvement.keys():
                dict_error_improvement[path] = (dict_error_improvement[
                                                    path] / total_error_improvement) * self.normalization_value
        return dict_error_improvement

    def correlation_importance(self, path_boost: 'SequentialPathBoost', variable_importance: dict) -> dict:

        # we want the ebm dataframe only for the paths tha have some importance
        frequency_name_of_the_paths = [ExtendedBoostingMatrix.generate_frequency_column_name_for_path(path_label=path)
                                       for path in variable_importance.keys()]
        train_ebm_dataframe = path_boost.train_ebm_dataframe_[frequency_name_of_the_paths]

        frequency_matrix = ExtendedBoostingMatrix.get_frequency_boosting_matrix(
            train_ebm_dataframe)

        # get the correlation matrix
        correlation_matrix = frequency_matrix.corr()

        # The keys in variable_importance and the correlation matrix must match.
        # variable_importance is a dict with tuple keys; correlation_matrix is a pd.DataFrame.

        correlation_variable_importance = dict()
        for path in variable_importance.keys():
            frequency_name_of_path = ExtendedBoostingMatrix.generate_frequency_column_name_for_path(path_label=path)
            correlation_variable_importance[path] = variable_importance[path]
            for second_path in variable_importance.keys():

                frequency_name_of_second_path = ExtendedBoostingMatrix.generate_frequency_column_name_for_path(
                    path_label=second_path)
                if len(path) > len(second_path) and path[:len(second_path)] == second_path:
                    corr = correlation_matrix.loc[frequency_name_of_path, frequency_name_of_second_path]
                    if not pd.isna(corr):
                        # we want to add the correlation only if it is not nan, it is nan
                        correlation_variable_importance[path] += corr * variable_importance[second_path]

        return correlation_variable_importance

    def combine_variable_importance_from_list_of_sequential_models(self, sequential_models: list, ) -> dict:

        variable_importance_dictionary = defaultdict(list)
        total_obs = 0
        for sequential_model in sequential_models:
            if sequential_model is not None:
                # we want to get the variable importance of each model and sum all them up
                n_obs = sequential_model.train_ebm_dataframe_.shape[0]
                total_obs += n_obs
                for key, value in sequential_model.variable_importance_.items():
                    variable_importance_dictionary[key].append(value * n_obs)

        averaged_variable_importance = {}
        for key, value_list in variable_importance_dictionary.items():
            if value_list:  # Check if the list is not empty to avoid division by zero
                averaged_variable_importance[key] = sum(value_list) / (len(value_list) * total_obs)
            else:
                averaged_variable_importance[key] = 0

        if self.normalize:
            total_error_improvement = sum(averaged_variable_importance.values())
            for path in averaged_variable_importance.keys():
                averaged_variable_importance[path] = (averaged_variable_importance[
                                                          path] / total_error_improvement) * self.normalization_value

        return averaged_variable_importance
