from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sequential_path_boost import SequentialPathBoost

import numpy as np


class VariableImportance:

    def __init__(self, criterion:str | None = None, use_correlation:bool = False):

        assert isinstance(criterion, str) or criterion is None

        self.criterion = criterion
        self.use_correlation = use_correlation

        criterion_choices = {
            'absolute': self.absolute_variable_importance,
            'relative': self.relative_variable_importance
        }

        # all the selected paths in order of selection
        self.iteration_numbers = []
        self.selected_path_at_iteration = []
        self.columns_at_iteration = []



    def _update(self, path_boost: 'SequentialPathBoost', selected_path: tuple, iteration_number: int):

        columns_names = path_boost.train_ebm_dataframe_.columns
        self.iteration_numbers.append(iteration_number)
        self.selected_path_at_iteration.append(selected_path)
        self.columns_at_iteration.append(columns_names)


        # check that the base learner has been already trained in this iteration
        assert (iteration_number == len(path_boost.base_learner_.train_mse) - 1)

        # check that the ebm has not been expanded yet
        assert path_boost._ebm_has_been_expanded_in_this_iteration is False




    def absolute_variable_importance(self, path_boost: 'SequentialPathBoost') -> float:
        # compute importance by error improvement
        # if we are in the first iteration there is no previous error to compare with
        # note: one can think to compare the first iteration with the error that we would have if we would have used
        # the mean of the labels (that is the variance of the labels in the case of MSE) however this does not work since
        # the bae learner is limited by the learining rate, sometimes making it first training even worse than the mean

        if iteration_number == 0:
            return 0

        # check that the iteration number is correct
        # if the iteration number does not coincide with the number of base learners (same as the number of errors)
        # it means we are in a new iteration, but we still have to train the base_learner


        for iteration in self.iteration_numbers:
            if iteration == 0:
                pass
            else:
                error_improvement = path_boost.base_learner_.train_mse[iteration] - path_boost.base_learner_.train_mse[iteration - 1]


        return error



    def relative_variable_importance(self, path_boost: 'SequentialPathBoost', selected_path: tuple,
                                     iteration_number: int):
    # it returns the improvment made by the selected tuple comparet to the improvement made by the second best tuple
    # this is a relative measure of importance


    # we assume the matrix has not been expanded yet

    # find second best path


