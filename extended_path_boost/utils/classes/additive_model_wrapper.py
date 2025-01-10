import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error
import numpy as np
from extended_boosting_matrix import ExtendedBoostingMatrix


class AdditiveModelWrapper:
    def __init__(self, BaseModelClass, base_model_class_kwargs, learning_rate: float, ):
        # model class will have an interface
        # TODO add the interface
        self.__last_train_prediction: pd.Series | None = None

        self.train_error = []
        self.eval_sets_mse = []
        self.learning_rate = learning_rate
        self.base_learners_list: list = []
        self.considered_columns = []
        self.BaseModelClass = BaseModelClass
        self.base_model_class_kwargs = base_model_class_kwargs

        # -----------------------------------------------------------------------------------------------------------
        self.base_learners_with_no_mean = []
        self.last_train_prediction_with_no_mean: pd.Series | None = None
        self.__last_test_prediction_with_no_mean: pd.Series | None = None
        self.train_error_with_no_mean = []
        self.test_error_with_no_mean = []

        self.base_learners_list_with_no_own_mean = []
        self.__last_train_prediction_with_no_own_mean: pd.Series | None = None
        self.__last_test_prediction_with_no_own_mean: pd.Series | None = None
        self.train_error_with_no_own_mean = []
        self.test_error_with_no_own_mean = []
        self.y_mean_with_no_own_mean_list = []

        self.base_learners_list_with_no_bl_mean = []
        self.__last_train_prediction_with_no_bl_mean: pd.Series | None = None
        self.__last_test_prediction_with_no_bl_mean: pd.Series | None = None
        self.train_error_with_no_bl_mean = []
        self.test_error_with_no_bl_mean = []

        # -----------------------------------------------------------------------------------------------------------

    def fit_one_step(self, X, y, best_path, eval_set=None, negative_gradient=None):
        # it fits one step of the boosting
        columns_to_keep = ExtendedBoostingMatrix.get_columns_related_to_path(best_path, X.columns)
        restricted_df = X[columns_to_keep]
        new_base_learner = self.BaseModelClass(**self.base_model_class_kwargs)



        self.trained_ = True
        if eval_set is not None:
            self.__last_eval_set_prediction_ = [pd.Series(np.zeros(len(ebm_df)), index=ebm_df.index) for ebm_df, y_eval in eval_set]
            

        if len(self.base_learners_list) == 0:
            # it is the first time we fit it so we do not need to compute the neg gradient


            new_base_learner.fit(restricted_df, y)
            self.base_learners_list.append(new_base_learner)
            self.considered_columns.append(columns_to_keep)
            base_learner_prediction = self.learning_rate * pd.Series(
                new_base_learner.predict(X[columns_to_keep]))
            self.__last_train_prediction = base_learner_prediction

            train_mse = mean_squared_error(y_true=y, y_pred=self.__last_train_prediction)
            self.train_error.append(train_mse)



        else:

            # compute the new target (we have to use zeroed_y - true_neg_gradient instead of just zeroed_y, more explained in paper)
            if negative_gradient is None:
                negative_gradient = self.__neg_gradient(y=y, y_hat=self.__last_train_prediction)
            new_y = pd.Series(negative_gradient)

            new_base_learner.fit(restricted_df, new_y)

            self.base_learners_list.append(new_base_learner)
            self.considered_columns.append(columns_to_keep)

            base_learner_prediction = self.learning_rate * new_base_learner.predict(X[columns_to_keep])
            self.__last_train_prediction += base_learner_prediction

            train_mse = mean_squared_error(y_true=y, y_pred=self.__last_train_prediction)
            self.train_error.append(train_mse)

           
      

        if eval_set is not None:
            eval_set_mse = []
            for i, ebm_df_eval, y_eval in enumerate(eval_set):
                assert isinstance(ebm_df_eval, pd.DataFrame)

                base_learner_prediction = self.learning_rate * pd.Series(
                    new_base_learner.predict(ebm_df_eval[columns_to_keep]))

                self.__last_eval_set_prediction_[i] +=  base_learner_prediction
                eval_set_mse.append(mean_squared_error(y_true=y_eval, y_pred=self.__last_eval_set_prediction_[i]))

            self.eval_sets_mse.append(eval_set_mse)
        
                
        return self

    def predict(self, X, **kwargs):
        prediction = []
        for i, base_learner in enumerate(self.base_learners_list):
            chosen_columns = self.considered_columns[i]
            base_learner_prediction = self.learning_rate * base_learner.predict(X[chosen_columns], **kwargs)
            prediction.append(base_learner_prediction)

        return sum(prediction)


    def evaluate(self, X, y, **kwargs):
        prediction = self.predict(X, **kwargs)
        mse = mean_squared_error(y_true=y, y_pred=prediction)
        return mse

    def get_model(self):
        return self.base_learners_list

    @staticmethod
    def __neg_gradient(y, y_hat):
        return y - y_hat
