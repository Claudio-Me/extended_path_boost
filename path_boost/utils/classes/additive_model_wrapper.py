import pandas as pd
import copy
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .extended_boosting_matrix import ExtendedBoostingMatrix
from typing import Iterable
from .interfaces.interface_base_learner import BaseLearnerClassInterface


class AdditiveModelWrapper:
    def __init__(self, BaseModelClass, base_model_class_kwargs, learning_rate: float,
                 learning_rate_scheduler: callable = None):
        """
        Initialize the AdditiveModelWrapper.

        Parameters
        ----------
        BaseModelClass : type
            The class of the base learner to use.
        base_model_class_kwargs : dict
            Keyword arguments to pass to the base learner constructor.
        learning_rate : float
            The initial learning rate.
        learning_rate_scheduler : callable, optional
            A function that takes `initial_lr` and `iteration` as arguments and returns
            the learning rate to use for that iteration. If None, learning rate is constant.
        """
        # Ensure BaseModelClass respects BaseLearnerClassInterface
        if not issubclass(BaseModelClass, BaseLearnerClassInterface):
            raise TypeError(f"{BaseModelClass.__name__} must implement BaseLearnerClassInterface")

        self._last_train_prediction: pd.Series | None = None

        self.train_mse = []
        self.train_mae = []
        self.eval_sets_mse: list[list[float]] = []
        self.eval_sets_mae: list[list[float]] = []
        self.learning_rate = learning_rate
        self._initial_learning_rate = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.base_learners_list: list = []
        self.considered_columns = []
        self.BaseModelClass = BaseModelClass
        self.base_model_class_kwargs = base_model_class_kwargs

    def fit_one_step(self, X: pd.DataFrame, y, best_path, eval_set=None, negative_gradient=None):
        # it fits one step of the boosting

        # Apply learning rate scheduler if provided
        if self.learning_rate_scheduler is not None:
            iteration = len(self.base_learners_list)
            self.learning_rate = self.learning_rate_scheduler(
                initial_lr=self._initial_learning_rate,
                iteration=iteration
            )

        columns_to_keep = ExtendedBoostingMatrix.get_columns_related_to_path(best_path, X.columns)
        restricted_df = X[columns_to_keep]
        if self.base_model_class_kwargs is not None:
            new_base_learner = self.BaseModelClass(**self.base_model_class_kwargs)
        else:
            new_base_learner = self.BaseModelClass()

        self.trained_ = True
        if eval_set is not None and not hasattr(self, '_last_eval_set_prediction_'):
            self._last_eval_set_prediction_ = []
            for eval_tuple in eval_set:
                if eval_tuple is None:
                    self._last_eval_set_prediction_.append(None)
                else:
                    self._last_eval_set_prediction_.append(
                        pd.Series(np.zeros(len(eval_tuple[0])), index=eval_tuple[0].index))

        if len(self.base_learners_list) == 0:
            # it is the first time we fit it so we do not need to compute the neg gradient

            self._target_variable_mean_ = []
            self._target_variable_mean_.append(np.array(y).mean())

            new_y = np.array(y) - self._target_variable_mean_[-1]

            new_base_learner.fit(restricted_df, new_y)
            self.base_learners_list.append(new_base_learner)
            self.considered_columns.append(columns_to_keep)
            base_learner_prediction = self._target_variable_mean_ + self.learning_rate * pd.Series(
                new_base_learner.predict(X[columns_to_keep]))
            self._last_train_prediction = base_learner_prediction

            train_mse = mean_squared_error(y_true=y, y_pred=self._last_train_prediction)
            train_mae = mean_absolute_error(y_true=y, y_pred=self._last_train_prediction)

            self.train_mse.append(train_mse)
            self.train_mae.append(train_mae)


        else:

            # compute the new target (we have to use zeroed_y - true_neg_gradient instead of just zeroed_y, more explained in paper)
            if negative_gradient is None:
                negative_gradient = self._neg_gradient(y=y, y_hat=self._last_train_prediction)
            new_y = np.array(negative_gradient)

            self._target_variable_mean_.append(new_y.mean())
            new_y = new_y - self._target_variable_mean_[-1]

            new_base_learner.fit(restricted_df, new_y)

            self.base_learners_list.append(new_base_learner)
            self.considered_columns.append(columns_to_keep)

            base_learner_prediction = self._target_variable_mean_[-1] + self.learning_rate * new_base_learner.predict(
                X[columns_to_keep])
            self._last_train_prediction += base_learner_prediction

            train_mse = mean_squared_error(y_true=y, y_pred=self._last_train_prediction)
            train_mae = mean_absolute_error(y_true=y, y_pred=self._last_train_prediction)

            self.train_mse.append(train_mse)
            self.train_mae.append(train_mae)

        if eval_set is not None:
            this_iter_eval_set_mse: list[float | None] = [None for _ in range(len(eval_set))]
            this_iter_eval_set_mae: list[float | None] = [None for _ in range(len(eval_set))]

            for i, eval_tuple in enumerate(eval_set):
                if eval_tuple is None:
                    self._last_eval_set_prediction_[i] = None
                    continue
                ebm_df_eval, y_eval = eval_tuple
                assert isinstance(ebm_df_eval, pd.DataFrame)

                base_learner_prediction = self._target_variable_mean_[
                                              -1] + self.learning_rate * new_base_learner.predict(
                    ebm_df_eval[columns_to_keep])

                self._last_eval_set_prediction_[i] += base_learner_prediction
                this_iter_eval_set_mse[i] = mean_squared_error(y_true=y_eval, y_pred=self._last_eval_set_prediction_[i])
                this_iter_eval_set_mae[i] = mean_absolute_error(y_true=y_eval,
                                                                y_pred=self._last_eval_set_prediction_[i])

            if len(self.eval_sets_mse) == 0:
                for eval_set_error in this_iter_eval_set_mse:
                    self.eval_sets_mse.append([eval_set_error])
            else:
                for i, eval_set_error in enumerate(this_iter_eval_set_mse):
                    self.eval_sets_mse[i].append(eval_set_error)

            if len(self.eval_sets_mae) == 0:
                for eval_set_error in this_iter_eval_set_mae:
                    self.eval_sets_mae.append([eval_set_error])
            else:
                for i, eval_set_error in enumerate(this_iter_eval_set_mae):
                    self.eval_sets_mae[i].append(eval_set_error)

        return self

    def predict(self, X: pd.DataFrame, **kwargs):
        predictions = self.predict_step_by_step(X, **kwargs)
        return predictions[-1]

    def predict_step_by_step(self, X: pd.DataFrame, **kwargs) -> list[np.array]:
        prediction = []
        last_prediction = np.zeros(len(X))
        for i, base_learner in enumerate(self.base_learners_list):
            chosen_columns = self.considered_columns[i]
            last_prediction += self._target_variable_mean_[i] + self.learning_rate * np.array(
                base_learner.predict(X[chosen_columns], **kwargs))
            prediction.append(copy.deepcopy(last_prediction))
        return prediction

    def evaluate(self, X: pd.DataFrame, y: Iterable, **kwargs) -> list[float]:
        # it returns the evolution of the mse with increasing number of iterations
        predictions = self.predict_step_by_step(X, **kwargs)
        evolution_mse = []
        for prediction in predictions:
            mse = mean_squared_error(y_true=y, y_pred=prediction)
            evolution_mse.append(mse)
        return evolution_mse

    def get_model(self):
        return self.base_learners_list

    @staticmethod
    def _neg_gradient(y, y_hat):
        return y - y_hat


# Learning rate schedulers
def exponential_decay_scheduler(initial_lr: float, iteration: int, decay_rate: float = 0.95) -> float:
    """
    Exponential decay learning rate scheduler.

    lr = initial_lr * decay_rate^iteration

    Parameters
    ----------
    initial_lr : float
        The initial learning rate.
    iteration : int
        The current iteration number (0-indexed).
    decay_rate : float, default=0.95
        The decay rate per iteration.

    Returns
    -------
    float
        The learning rate for this iteration.
    """
    return initial_lr * (decay_rate ** iteration)


def step_decay_scheduler(initial_lr: float, iteration: int, drop_every: int = 10, drop_factor: float = 0.5) -> float:
    """
    Step decay learning rate scheduler.

    Learning rate drops by drop_factor every drop_every iterations.

    Parameters
    ----------
    initial_lr : float
        The initial learning rate.
    iteration : int
        The current iteration number (0-indexed).
    drop_every : int, default=10
        Number of iterations between drops.
    drop_factor : float, default=0.5
        Factor by which to multiply the learning rate at each drop.

    Returns
    -------
    float
        The learning rate for this iteration.
    """
    return initial_lr * (drop_factor ** (iteration // drop_every))


def linear_decay_scheduler(initial_lr: float, iteration: int, total_iterations: int = 100, min_lr: float = 0.01) -> float:
    """
    Linear decay learning rate scheduler.

    Learning rate decays linearly from initial_lr to min_lr over total_iterations.

    Parameters
    ----------
    initial_lr : float
        The initial learning rate.
    iteration : int
        The current iteration number (0-indexed).
    total_iterations : int, default=100
        Total number of iterations over which to decay.
    min_lr : float, default=0.01
        Minimum learning rate.

    Returns
    -------
    float
        The learning rate for this iteration.
    """
    decay = (initial_lr - min_lr) / total_iterations
    return max(min_lr, initial_lr - decay * iteration)
