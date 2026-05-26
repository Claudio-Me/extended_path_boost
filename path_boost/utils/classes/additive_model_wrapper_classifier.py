import pandas as pd
import copy
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score
from .extended_boosting_matrix import ExtendedBoostingMatrix
from typing import Iterable
from .interfaces.interface_base_learner import BaseLearnerClassInterface
from sklearn.tree import DecisionTreeRegressor


class AdditiveModelWrapperClassifier:
    def __init__(
        self,
        BaseModelClass,
        base_model_class_kwargs,
        learning_rate: float,
        use_tree_boost: bool = False,
    ):
        self.use_tree_boost = use_tree_boost
        if self.use_tree_boost:
            BaseModelClass = BoostedTreeBaselearner

        # Ensure BaseModelClass respects BaseLearnerClassInterface
        if not issubclass(BaseModelClass, BaseLearnerClassInterface):
            raise TypeError(
                f"{BaseModelClass.__name__} must implement BaseLearnerClassInterface"
            )

        self._last_train_prediction: pd.Series | None = None

        self.train_logloss = []
        self.train_accuracy = []
        self.eval_sets_logloss: list[list[float]] = []
        self.eval_sets_accuracy: list[list[float]] = []
        self.learning_rate = learning_rate
        self.base_learners_list: list[BaseLearnerClassInterface] = []
        self.considered_columns = []
        self.BaseModelClass = BaseModelClass
        self.base_model_class_kwargs = base_model_class_kwargs

    def fit_one_step(self, X: pd.DataFrame, y, best_path, eval_set=None):
        # it fits one step of the boosting

        columns_to_keep = ExtendedBoostingMatrix.get_columns_related_to_path(
            best_path, X.columns
        )
        restricted_df = X[columns_to_keep]

        self.trained_ = True
        if eval_set is not None and not hasattr(self, "_last_eval_set_prediction_"):
            self._last_eval_set_prediction_ = []
            for eval_tuple in eval_set:
                if eval_tuple is None:
                    self._last_eval_set_prediction_.append(None)
                else:
                    self._last_eval_set_prediction_.append(
                        pd.Series(
                            np.zeros(len(eval_tuple[0])), index=eval_tuple[0].index
                        )
                    )

        if len(self.base_learners_list) == 0:
            # it is the first time we fit it so we do not need to compute the neg gradient
            new_base_learner = FirstConstantBaseLearner()
            self._target_variable_mean_ = []
            self._target_variable_mean_.append(0.0)

            new_base_learner.fit(restricted_df, np.array(y))
            self.base_learners_list.append(new_base_learner)
            self.considered_columns.append(columns_to_keep)

            # this gives the log-odd of being in class 1 (F(x))
            self._last_train_model_output_F = pd.Series(
                new_base_learner.predict(X[columns_to_keep])
            )

        else:
            # compute the new target (we have to use zeroed_y - true_neg_gradient instead of just zeroed_y, more explained in paper)

            if self.base_model_class_kwargs is not None:
                new_base_learner = self.BaseModelClass(**self.base_model_class_kwargs)
            else:
                new_base_learner = self.BaseModelClass()

            negative_gradient = self._neg_gradient(
                y=y, y_hat=self._last_train_prediction_probability
            )
            new_y = np.array(negative_gradient)

            self._target_variable_mean_.append(new_y.mean())
            new_y = new_y - self._target_variable_mean_[-1]

            if self.use_tree_boost:
                new_base_learner.fit(
                    restricted_df,
                    neg_gradient=new_y,
                    current_f=self._last_train_model_output_F.values,
                    y_true=np.array(y),
                )
            else:
                new_base_learner.fit(restricted_df, new_y)

            self.base_learners_list.append(new_base_learner)
            self.considered_columns.append(columns_to_keep)

            last_train_model_output_F = self._target_variable_mean_[
                -1
            ] + self.learning_rate * self.base_learners_list[-1].predict(
                X[columns_to_keep]
            )
            self._last_train_model_output_F += last_train_model_output_F

        # we transform the model prediction F into probability (p(x) or y_hat)
        self._last_train_prediction_probability = self._sigmoid(
            self._last_train_model_output_F
        )

        # this gives us the predicted class
        self._last_train_prediction = (
            self._last_train_prediction_probability >= 0.5
        ).astype(int)

        train_logloss = log_loss(
            y_true=y, y_pred=self._last_train_prediction_probability
        )
        train_accuracy = accuracy_score(y_true=y, y_pred=self._last_train_prediction)

        self.train_logloss.append(train_logloss)
        self.train_accuracy.append(train_accuracy)

        if eval_set is not None:
            this_iter_eval_set_logloss: list[float | None] = [
                None for _ in range(len(eval_set))
            ]
            this_iter_eval_set_accuracy: list[float | None] = [
                None for _ in range(len(eval_set))
            ]

            for i, eval_tuple in enumerate(eval_set):
                if eval_tuple is None:
                    self._last_eval_set_prediction_[i] = None
                    continue
                ebm_df_eval, y_eval = eval_tuple
                assert isinstance(ebm_df_eval, pd.DataFrame)

                base_learner_prediction_F = self._target_variable_mean_[
                    -1
                ] + self.learning_rate * new_base_learner.predict(
                    ebm_df_eval[columns_to_keep]
                )

                self._last_eval_set_prediction_[i] += base_learner_prediction_F

                # we transform the model prediction F into probability (p(x) or y_hat)
                last_eval_set_prediction_probability = self._sigmoid(
                    self._last_eval_set_prediction_[i]
                )

                # this gives us the predicted class
                self._last_eval_set_prediction = (
                    last_eval_set_prediction_probability >= 0.5
                ).astype(int)

                eval_logloss = log_loss(
                    y_true=y_eval,
                    y_pred=last_eval_set_prediction_probability,
                    labels=[0, 1],
                )
                eval_accuracy = accuracy_score(
                    y_true=y_eval, y_pred=self._last_eval_set_prediction
                )

                this_iter_eval_set_logloss[i] = eval_logloss
                this_iter_eval_set_accuracy[i] = eval_accuracy

            if len(self.eval_sets_logloss) == 0:
                for eval_set_error in this_iter_eval_set_logloss:
                    self.eval_sets_logloss.append([eval_set_error])
            else:
                for i, eval_set_error in enumerate(this_iter_eval_set_logloss):
                    self.eval_sets_logloss[i].append(eval_set_error)

            if len(self.eval_sets_accuracy) == 0:
                for eval_set_error in this_iter_eval_set_accuracy:
                    self.eval_sets_accuracy.append([eval_set_error])
            else:
                for i, eval_set_error in enumerate(this_iter_eval_set_accuracy):
                    self.eval_sets_accuracy[i].append(eval_set_error)

        return self

    def predict(self, X: pd.DataFrame, class_probability: bool = False, **kwargs):
        predictions = self.predict_step_by_step(
            X, return_class_probability=class_probability, **kwargs
        )
        return predictions[-1]

    def predict_step_by_step(
        self, X: pd.DataFrame, return_class_probability=False, **kwargs
    ) -> list[np.array]:
        """
        Generates predictions for each boosting iteration step.

        Args:
            X (pd.DataFrame): Input features for prediction.
            return_class_probability (bool, optional): If True, returns class probabilities for each step.
                If False, returns binary class predictions. Defaults to False.
            **kwargs: Additional keyword arguments passed to the base learner's predict method.

        Returns:
            list[np.array]: List of predictions for each boosting step. Each element is either
                an array of class probabilities or binary predictions, depending on `return_class_probability`.
        """
        prediction = []
        last_prediction_model_F = np.zeros(len(X))
        for i, base_learner in enumerate(self.base_learners_list):
            chosen_columns = self.considered_columns[i]
            # the first base learner is not scaled by the learning rate because it is just the average of the labels
            if i == 0:
                learning_rate = 1.0
            else:
                learning_rate = self.learning_rate
            last_prediction_model_F += self._target_variable_mean_[
                i
            ] + learning_rate * np.array(
                base_learner.predict(X[chosen_columns], **kwargs)
            )

            last_prediction_probability = self._sigmoid(last_prediction_model_F)
            if return_class_probability:
                prediction.append(last_prediction_probability)
            else:
                prediction.append((last_prediction_probability >= 0.5).astype(int))

        return prediction

    def evaluate(self, X: pd.DataFrame, y: Iterable, **kwargs) -> list[float]:
        # it returns the evolution of the mse with increasing number of iterations
        predictions = self.predict_step_by_step(
            X, return_class_probability=True, **kwargs
        )
        evolution_logloss = []
        for prediction in predictions:
            logloss = log_loss(y_true=y, y_pred=prediction)
            evolution_logloss.append(logloss)
        return evolution_logloss

    def get_model(self):
        return self.base_learners_list

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _neg_gradient(y, y_hat):
        return y - y_hat


class BoostedTreeBaselearner(BaseLearnerClassInterface):
    def __init__(self, **kwargs):
        self.model = DecisionTreeRegressor(**kwargs)
        self.fitted_ = False
        self.leaf_gammas = {}  # Store optimized gamma for each leaf

    def fit(
        self,
        X: pd.DataFrame,
        neg_gradient: Iterable,
        current_f: np.ndarray = None,
        y_true: np.ndarray = None,
        **kwargs,
    ):
        """
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        neg_gradient : Iterable
            Pseudo-residuals (negative gradient)
        current_f : np.ndarray
            Current model predictions (in log-odds space)
        y_true : np.ndarray
            True labels in {0, 1} format
        """
        # First fit tree to pseudo-residuals
        self.model.fit(X, neg_gradient)

        # If we have current_f and y_true, optimize gamma for each leaf
        if current_f is not None and y_true is not None:
            self._optimize_leaf_gammas(X, y_true, current_f)
        else:
            # Fallback: use tree's predictions as-is
            leaf_indices = self.model.apply(X)
            unique_leaves = np.unique(leaf_indices)
            for leaf_id in unique_leaves:
                mask = leaf_indices == leaf_id
                # Use average of pseudo-residuals in leaf
                self.leaf_gammas[leaf_id] = self.model.predict(
                    X.iloc[[np.where(mask)[0][0]]]
                )[0]

        self.fitted_ = True
        return self

    def _optimize_leaf_gammas(
        self, X: pd.DataFrame, y_true: np.ndarray, current_f: np.ndarray
    ):
        """
        For each leaf, find optimal gamma that minimizes logistic loss
        Works with y_true in {0, 1}
        """
        # Get leaf assignments for all samples
        leaf_indices = self.model.apply(X)
        unique_leaves = np.unique(leaf_indices)

        for leaf_id in unique_leaves:
            # Get samples in this leaf
            mask = leaf_indices == leaf_id

            if np.sum(mask) == 0:
                self.leaf_gammas[leaf_id] = 0.0
                continue

            # Get tree's base prediction for this leaf (all same value)
            X_leaf = X[mask]
            h_pred_leaf = self.model.predict(X_leaf.iloc[[0]])[0]

            # Get data for this leaf
            y_leaf = y_true[mask]  # Shape: (n_samples_in_leaf,), values in {0, 1}
            f_leaf = current_f[mask]  # Shape: (n_samples_in_leaf,), log-odds

            # Optimize gamma for this specific leaf
            def loss(gamma):
                f_new = f_leaf + gamma * h_pred_leaf
                # Binary cross-entropy (logistic loss) for y in {0, 1}:
                # -[y*log(p) + (1-y)*log(1-p)] where p = sigmoid(f)
                # Equivalent to: log(1 + exp(-f)) if y=1, log(1 + exp(f)) if y=0
                # Combined: y*log(1 + exp(-f)) + (1-y)*log(1 + exp(f))
                p = 1 / (1 + np.exp(-np.clip(f_new, -500, 500)))  # sigmoid
                return -np.sum(
                    y_leaf * np.log(p + 1e-15) + (1 - y_leaf) * np.log(1 - p + 1e-15)
                )

            # Simple line search
            best_gamma = 0.0
            best_loss = loss(0.0)

            # Search in reasonable range
            for gamma in np.linspace(-10, 10, 100):
                current_loss = loss(gamma)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_gamma = gamma

            self.leaf_gammas[leaf_id] = best_gamma

    def predict(self, X: pd.DataFrame, **kwargs):
        """
        Predict using optimized gamma values instead of tree's leaf values
        """
        if not self.fitted_:
            raise ValueError("Model not fitted yet")

        # Get leaf assignments
        leaf_indices = self.model.apply(X)

        # Map each sample to its leaf's optimized gamma
        predictions = np.array(
            [self.leaf_gammas.get(leaf_id, 0.0) for leaf_id in leaf_indices]
        )

        return predictions


class FirstConstantBaseLearner(BaseLearnerClassInterface):
    # use for the first base learner in classification tasks
    # it always predicts the most frequent class in the training set
    def __init__(self):
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: Iterable, **kwargs):
        self.fitted_ = True
        self.unique_classes_ = np.unique(y)
        class_mean = np.mean(y)
        self._predict_value = self._log_odds(class_mean)
        return self

    def predict(self, X: pd.DataFrame, **kwargs):
        return self._predict_value * np.ones(len(X))

    def _log_odds(self, mean_y):
        return np.log(mean_y / (1 - mean_y))
