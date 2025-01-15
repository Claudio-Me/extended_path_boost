import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error
import numpy as np
from .extended_boosting_matrix import ExtendedBoostingMatrix
from typing import Iterable
import copy
from sklearn import tree
import matplotlib.pyplot as plt


class AdditiveModelWrapper:
    def __init__(self, BaseModelClass, base_model_class_kwargs, learning_rate: float, ):
        # model class will have an interface
        # TODO add the interface

        self._last_train_prediction: pd.Series | None = None

        self.train_mse = []
        self.eval_sets_mse: list[list[float]] = []
        self.learning_rate = learning_rate
        self.base_learners_list: list = []
        self.considered_columns = []
        self.BaseModelClass = BaseModelClass
        self.base_model_class_kwargs = base_model_class_kwargs

    def fit_one_step(self, X: pd.DataFrame, y, best_path, eval_set=None, negative_gradient=None):
        # it fits one step of the boosting

        columns_to_keep = ExtendedBoostingMatrix.get_columns_related_to_path(best_path, X.columns)
        restricted_df = X[columns_to_keep]
        new_base_learner = self.BaseModelClass(**self.base_model_class_kwargs)

        self.trained_ = True
        if eval_set is not None and not hasattr(self, '_last_eval_set_prediction_'):
            self._last_eval_set_prediction_ = [pd.Series(np.zeros(len(ebm_df)), index=ebm_df.index) for ebm_df, _
                                               in eval_set]

        if len(self.base_learners_list) == 0:
            # it is the first time we fit it so we do not need to compute the neg gradient

            new_base_learner.fit(restricted_df, y)
            self.base_learners_list.append(new_base_learner)
            self.considered_columns.append(columns_to_keep)
            base_learner_prediction = self.learning_rate * pd.Series(
                new_base_learner.predict(X[columns_to_keep]))
            self._last_train_prediction = base_learner_prediction

            train_mse = mean_squared_error(y_true=y, y_pred=self._last_train_prediction)
            self.train_mse.append(train_mse)



        else:

            # compute the new target (we have to use zeroed_y - true_neg_gradient instead of just zeroed_y, more explained in paper)
            if negative_gradient is None:
                negative_gradient = self._neg_gradient(y=y, y_hat=self._last_train_prediction)
            new_y = pd.Series(negative_gradient)

            new_base_learner.fit(restricted_df, new_y)

            # ----------------------------------------------------------------------------------------
            # debugging
            # Plot the tree
            #plt.figure(figsize=(12, 8))
            #tree.plot_tree(new_base_learner, filled=True, feature_namesTrue)
            #plt.show()
            # ----------------------------------------------------------------------------------------

            self.base_learners_list.append(new_base_learner)
            self.considered_columns.append(columns_to_keep)

            base_learner_prediction = self.learning_rate * new_base_learner.predict(X[columns_to_keep])
            self._last_train_prediction += base_learner_prediction

            train_mse = mean_squared_error(y_true=y, y_pred=self._last_train_prediction)
            self.train_mse.append(train_mse)

        if eval_set is not None:
            eval_set_mse = []
            for i, eval_tuple in enumerate(eval_set):
                ebm_df_eval, y_eval = eval_tuple
                assert isinstance(ebm_df_eval, pd.DataFrame)

                base_learner_prediction = self.learning_rate * new_base_learner.predict(ebm_df_eval[columns_to_keep])

                self._last_eval_set_prediction_[i] += base_learner_prediction
                eval_set_mse.append(mean_squared_error(y_true=y_eval, y_pred=self._last_eval_set_prediction_[i]))

            self.eval_sets_mse.append(eval_set_mse)

        return self

    def predict(self, X: pd.DataFrame, **kwargs):
        predictions = self._predictions_step_by_step(X, **kwargs)
        return predictions[-1]

    def _predictions_step_by_step(self, X: pd.DataFrame, **kwargs) -> list[np.array]:
        prediction = []
        last_prediction= np.zeros(len(X))
        for i, base_learner in enumerate(self.base_learners_list):
            chosen_columns = self.considered_columns[i]
            last_prediction += self.learning_rate * np.array(base_learner.predict(X[chosen_columns], **kwargs))
            prediction.append(copy.deepcopy(last_prediction))
        return prediction

    def evaluate(self, X: pd.DataFrame, y: Iterable, **kwargs):
        # it returns the evolution of the mse with increasing number of iterations
        predictions = self._predictions_step_by_step(X, **kwargs)
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
