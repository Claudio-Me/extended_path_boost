import networkx as nx
import pandas as pd
import numbers
import matplotlib.pyplot as plt
import numpy as np
import logging

from .interfaces.interface_base_learner import BaseLearnerClassInterface
from .interfaces.interface_selector import SelectorClassInterface
from ..validate_data import util_validate_data
from ..variable_importance_according_to_path_boost import VariableImportance_ForSequentialPathBoost
from ..plots_functions import plot_training_and_eval_errors, plot_variable_importance_utils

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from .extended_boosting_matrix import ExtendedBoostingMatrix
from typing import Iterable, Union, Optional, List, Tuple, Dict, Any, Type, Callable
from sklearn.tree import DecisionTreeRegressor, plot_tree
from .additive_model_wrapper import AdditiveModelWrapper
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import MaxNLocator

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Set up logger for the module
logger = logging.getLogger('path_boost')

# Type aliases
GraphList = List[nx.Graph]
PathTuple = Tuple[Union[int, str], ...]
EvalSet = List[Tuple[GraphList, Iterable]]


class SequentialPathBoost(BaseEstimator, RegressorMixin):
    """
    Gradient boosting for graph-structured data using path-based features.

    SequentialPathBoost iteratively discovers labeled paths in graphs that
    are predictive of the target variable. At each iteration, it:

    1. Selects the most informative path using a selector model
    2. Expands the Extended Boosting Matrix (EBM) with path extensions
    3. Fits a base learner on the selected path's features
    4. Updates predictions using gradient boosting

    This class is typically used through PathBoost, which handles multiple
    anchor node types in parallel.

    Attributes
    ----------
    train_ebm_dataframe_ : pd.DataFrame
        The Extended Boosting Matrix built during training.
    train_mse_ : List[float]
        Training MSE at each iteration.
    paths_selected_by_epb_ : set
        Set of paths selected during training.
    variable_importance_ : Dict[str, float]
        Path importance scores (if parameters_variable_importance provided).
    is_fitted_ : bool
        Whether the model has been fitted.

    See Also
    --------
    PathBoost : Ensemble variant with multiple anchor node types.
    """

    # Fitted attributes (declared for type checking)
    train_ebm_dataframe_: pd.DataFrame
    train_mse_: List[float]
    train_mae_: List[float]
    paths_selected_by_epb_: set
    variable_importance_: Optional[Dict[str, float]]
    is_fitted_: bool
    columns_names_: List[str]

    def __init__(
        self,
        n_iter: int = 100,
        max_path_length: int = 10,
        learning_rate: float = 0.1,
        patience: Optional[int] = None,
        target_error: Optional[float] = None,
        tol: float = 1e-4,
        restore_best_model: bool = True,
        learning_rate_scheduler: Optional[Callable[[float, int], float]] = None,
        BaseLearnerClass: type = DecisionTreeRegressor,
        kwargs_for_base_learner: Optional[Dict[str, Any]] = None,
        SelectorClass: type = DecisionTreeRegressor,
        kwargs_for_selector: Optional[Dict[str, Any]] = None,
        parameters_variable_importance: Optional[Dict[str, Any]] = None,
        replace_nan_with: float = np.nan,
        verbose: bool = False
    ) -> None:
        """
              Initializes the SequentialPathBoost model.

              Parameters
              ----------
              n_iter : int, default=100
                  The number of boosting iterations to perform.
              max_path_length : int, default=10
                  The maximum length of paths to consider as features. Paths longer
                  than this will not be explored for extending the Extended Boosting Matrix (EBM).
              learning_rate : float, default=0.1
                  The learning_rate shrinks the contribution of each base learner.
                  It is used by the `AdditiveModelWrapper` when fitting each step.
              patience : int, optional, default=None
                  Number of iterations with no improvement on the first evaluation set's score
                  before stopping early. If None, early stopping is not performed.
                  Requires an `eval_set` to be provided during fitting. The check is performed
                  based on the Mean Squared Error (MSE) of the first evaluation set in `eval_set`.
              tol : float, default=1e-4
                  Minimum improvement in evaluation MSE required to consider as "improvement".
                  If the MSE decreases by less than `tol` over `patience` iterations, training stops.
                  This prevents stopping on insignificant improvements.
              restore_best_model : bool, default=True
                  If True and eval_set is provided, the model will be restored to the iteration
                  with the lowest evaluation MSE after training completes. This prevents returning
                  an overfit model when early stopping doesn't trigger soon enough.
              learning_rate_scheduler : callable, optional, default=None
                  A function that takes `initial_lr` and `iteration` as arguments and returns
                  the learning rate to use for that iteration. If None, the learning rate is constant.
                  Built-in schedulers: `exponential_decay_scheduler`, `step_decay_scheduler`, `linear_decay_scheduler`.
              BaseLearnerClass : type, default=sklearn.tree.DecisionTreeRegressor
                  The class of the base learner to be used within each boosting iteration.
                  This class must implement the `BaseLearnerClassInterface`.
              kwargs_for_base_learner : dict, default=None
                  Keyword arguments to be passed to the constructor of the `BaseLearnerClass`.
                  If None, default arguments for `DecisionTreeRegressor` will be used.
              SelectorClass : type, default=sklearn.tree.DecisionTreeRegressor
                  The class of the feature selector used to identify the best paths in each iteration.
                  This class must implement the `SelectorClassInterface`.
              kwargs_for_selector : dict, default=None
                  Keyword arguments to be passed to the constructor of the `SelectorClass`.
                  If None, default arguments for `DecisionTreeRegressor` will be used.
              parameters_variable_importance : dict, default=None
                  Parameters for computing variable importance. If None, variable importance is not computed.
                  Expected keys include 'criterion', 'error_used', 'use_correlation', 'normalize'.
              replace_nan_with : any, default=np.nan
                  Value used to replace NaN values encountered during feature generation in the EBM.
                  This is important for base learners that cannot handle NaN values.
              verbose : bool, default=False
                  If True, prints progress messages during the fitting process, such as the
                  current iteration number and the best path selected.
        """
        self.n_iter = n_iter
        self.max_path_length = max_path_length
        self.patience = patience
        self.target_error = target_error
        self.tol = tol
        self.restore_best_model = restore_best_model
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate = learning_rate
        self.BaseLearnerClass = BaseLearnerClass
        self.verbose = verbose
        self.replace_nan_with = replace_nan_with
        self.kwargs_for_base_learner = kwargs_for_base_learner
        self.SelectorClass = SelectorClass
        self.kwargs_for_selector = kwargs_for_selector
        self.parameters_variable_importance = parameters_variable_importance

    def fit(self, X: list[nx.Graph], y: np.array, list_anchor_nodes_labels: list[tuple], anchor_nodes_label_name,
            eval_set: list[tuple[list[nx.Graph], Iterable]] = None):
        """
        Fits the SequentialPathBoost model to the training data.

        This method iteratively builds an ensemble of base learners. In each iteration:
        1. It identifies the 'best path' from the current set of available paths in the
           Extended Boosting Matrix (EBM) using a selector model. The target for the
           selector is the original target `y` in the first iteration, and the
           negative gradient of the loss function in subsequent iterations.
        2. It trains a new base learner on the features corresponding to the `best_path`
           and adds it to the ensemble. The `AdditiveModelWrapper` handles the
           fitting of this base learner and updates the cumulative predictions.
        3. It expands the training EBM by generating new path-based features derived
           from extending the `best_path`.
        4. If variable importance calculation is enabled, it updates the importance scores
           based on the selected path and the current gradient.
        5. It expands the EBM for evaluation sets (if provided) to include features
           derived from the `best_path`.

        The process continues for `n_iter` iterations. After fitting, training and
        evaluation (if `eval_set` is provided) metrics (MSE, MAE) are stored.
        If `parameters_variable_importance` was set, the final variable importance
        scores are computed.

        Parameters
        ----------
        X : list[nx.Graph]
            A list of NetworkX graph objects representing the training input samples.
        y : np.array
            A NumPy array of target values corresponding to `X`.
        list_anchor_nodes_labels : list[tuple]
            A list of tuples, where each tuple contains the label(s) identifying
            anchor nodes. These are used to initialize the EBM.
        anchor_nodes_label_name : str
            The name of the node attribute in the graphs that contains the labels
            used to identify anchor nodes and subsequent path elements.
        eval_set : list[tuple[list[nx.Graph], Iterable]], optional, default=None
            A list of (X_eval, y_eval) tuples for monitoring the model's performance
            on one or more evaluation sets during training.

        Returns
        -------
        self : object
            The fitted SequentialPathBoost estimator.
        """

        self._default_kwargs_for_base_learner = {'max_depth': 3,
                                                 'random_state': 0,
                                                 'splitter': 'best',
                                                 }

        self._default_kwargs_for_selector = {'max_depth': 1,
                                             'random_state': 0,
                                             'splitter': 'best',
                                             'criterion': "squared_error"
                                             }

        self._validate_data(X=X, y=y, list_anchor_nodes_labels=list_anchor_nodes_labels,
                            name_of_label_attribute=anchor_nodes_label_name, eval_set=eval_set,
                            parameters_variable_importance=self.parameters_variable_importance,
                            patience=self.patience)

        self.is_fitted_ = True

        self.name_of_label_attribute_ = anchor_nodes_label_name

        self.paths_selected_by_epb_ = set()
        self._initialize_path_boosting(X=X,
                                       list_anchor_nodes_labels=list_anchor_nodes_labels,
                                       main_label_name=anchor_nodes_label_name,
                                       eval_set=eval_set)

        if self.parameters_variable_importance is not None:
            self.class_variable_importance_: VariableImportance_ForSequentialPathBoost = VariableImportance_ForSequentialPathBoost(
                **self.parameters_variable_importance)

        # Log training start
        logger.info(f"Starting SequentialPathBoost training with {len(X)} samples, {self.n_iter} max iterations")
        logger.debug(f"Training parameters: learning_rate={self.learning_rate}, max_path_length={self.max_path_length}")

        # Set up iterator with optional progress bar
        iterator = range(self.n_iter)
        if self.verbose and TQDM_AVAILABLE:
            iterator = tqdm(iterator, desc="SequentialPathBoost", unit="iter",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for n_iteration in iterator:
            if self.verbose and not TQDM_AVAILABLE:
                print("iteration number: ", n_iteration + 1)
            logger.debug(f"Starting iteration {n_iteration + 1}")

            # this is a parameter used for a check when computing variable importance, to make sure we are computing it on the right iteration, with the right ebm
            self._ebm_has_been_expanded_in_this_iteration = False

            if n_iteration == 0:
                best_path = self._find_best_path(train_ebm_dataframe=self.train_ebm_dataframe_,
                                                 y=y,
                                                 SelectorClass=self.SelectorClass,
                                                 kwargs_for_selector=self.kwargs_for_selector)
            else:

                negative_gradient = AdditiveModelWrapper._neg_gradient(y=y, y_hat=np.array(
                    self.base_learner_._last_train_prediction.to_numpy()))
                best_path = self._find_best_path(train_ebm_dataframe=self.train_ebm_dataframe_,
                                                 y=pd.Series(negative_gradient),
                                                 SelectorClass=self.SelectorClass,
                                                 kwargs_for_selector=self.kwargs_for_selector)

            # Log selected path
            logger.debug(f"Iteration {n_iteration + 1}: selected path {best_path}")

            if self.verbose:
                if TQDM_AVAILABLE:
                    # Update tqdm postfix with best path info
                    if hasattr(iterator, 'set_postfix'):
                        iterator.set_postfix({'path': str(best_path)[:30]})
                else:
                    print("Best path: ", best_path)

            # we collect some values for variable importance, important that this operation it is done between the
            # selection of the best path and the expansion of the ebm dataframe
            #-------------------------------------------------------------------------------------------------
            if self.parameters_variable_importance is not None:
                if n_iteration == 0:
                    self.class_variable_importance_._update(path_boost=self, selected_path=best_path,
                                                            iteration_number=n_iteration, gradient=y)
                else:
                    self.class_variable_importance_._update(path_boost=self, selected_path=best_path,
                                                            iteration_number=n_iteration, gradient=negative_gradient)

            #-------------------------------------------------------------------------------------------------

            # expand the EVAL set in order to contain the selected columns path
            self._expand_eval_ebm_dataframe_with_best_path(best_path=best_path, anchor_node_label_name=anchor_nodes_label_name,
                                                           eval_set=eval_set)

            self.base_learner_.fit_one_step(X=self.train_ebm_dataframe_, y=y, best_path=best_path,
                                            eval_set=self.eval_set_ebm_df_and_target_)

            if eval_set is not None:
                # Track best model for potential restoration
                if len(self.base_learner_.eval_sets_mse) > 0 and len(self.base_learner_.eval_sets_mse[0]) > 0:
                    current_eval_mse = self.base_learner_.eval_sets_mse[0][-1]
                    if not hasattr(self, '_best_eval_mse_') or current_eval_mse < self._best_eval_mse_:
                        self._best_eval_mse_ = current_eval_mse
                        self._best_iteration_ = n_iteration
                        self._best_base_learners_count_ = len(self.base_learner_.base_learners_list)

                if self._check_if_stop_early(mse_eval_set=self.base_learner_.eval_sets_mse[0], patience=self.patience,
                                             target_error=self.target_error, tol=self.tol):
                    logger.info(f"Early stopping at iteration {n_iteration + 1} (best MSE: {self._best_eval_mse_:.6f})")
                    if self.verbose:
                        print(
                            f"Early stopping at iteration {n_iteration + 1} due to no improvement in evaluation set MSE.")
                        self.n_iter = n_iteration
                    break

            # expand the ebm dataframe with the new columns starting from the selected path
            self._expand_ebm_dataframe(X=X, selected_path=best_path, main_label_name=anchor_nodes_label_name)

        self.train_mse_ = self.base_learner_.train_mse
        self.train_mae_ = self.base_learner_.train_mae

        if self.parameters_variable_importance is not None:
            self.variable_importance_: dict = self.class_variable_importance_.compute_variable_importance(
                path_boost=self)

        if eval_set is not None:
            self.eval_sets_mse_ = self.base_learner_.eval_sets_mse
            self.eval_sets_mae_ = self.base_learner_.eval_sets_mae

        self.columns_names_ = self.train_ebm_dataframe_.columns

        # Restore best model if enabled and we have a best iteration
        if self.restore_best_model and eval_set is not None and hasattr(self, '_best_iteration_'):
            final_iteration = len(self.base_learner_.base_learners_list) - 1
            if self._best_iteration_ < final_iteration:
                # Truncate to best iteration
                self.base_learner_.base_learners_list = self.base_learner_.base_learners_list[:self._best_base_learners_count_]
                self.base_learner_.considered_columns = self.base_learner_.considered_columns[:self._best_base_learners_count_]
                self.base_learner_._target_variable_mean_ = self.base_learner_._target_variable_mean_[:self._best_base_learners_count_]
                self.base_learner_.train_mse = self.base_learner_.train_mse[:self._best_base_learners_count_]
                self.base_learner_.train_mae = self.base_learner_.train_mae[:self._best_base_learners_count_]
                # Also truncate eval set metrics
                for i in range(len(self.base_learner_.eval_sets_mse)):
                    self.base_learner_.eval_sets_mse[i] = self.base_learner_.eval_sets_mse[i][:self._best_base_learners_count_]
                    self.base_learner_.eval_sets_mae[i] = self.base_learner_.eval_sets_mae[i][:self._best_base_learners_count_]
                # Update the stored metrics
                self.train_mse_ = self.base_learner_.train_mse
                self.train_mae_ = self.base_learner_.train_mae
                self.eval_sets_mse_ = self.base_learner_.eval_sets_mse
                self.eval_sets_mae_ = self.base_learner_.eval_sets_mae

                if self.verbose:
                    print(f"Restored model to best iteration {self._best_iteration_ + 1} with eval MSE {self._best_eval_mse_:.6f}")
                logger.info(f"Restored model to best iteration {self._best_iteration_ + 1}")

        # Log training completion
        final_mse = self.train_mse_[-1] if len(self.train_mse_) > 0 else None
        logger.info(f"Training completed: {len(self.train_mse_)} iterations, final train MSE: {final_mse}")

        return self

    def _check_if_stop_early(self, mse_eval_set: list[float], patience: int | None = None,
                             target_error: float | None = None, tol: float = 1e-4) -> bool:
        """
        Determines whether to stop the training process early based on evaluation metrics.

        Early stopping can be triggered under two conditions:
        1. If a `target_error` is specified: Training stops if the Mean Squared Error (MSE)
           on the (first) evaluation set falls at or below this target.
        2. If `patience` is specified: Training stops if the MSE on the (first) evaluation
           set has not improved by at least `tol` over `patience` iterations. This prevents
           stopping on insignificant improvements.

        Parameters
        ----------
        mse_eval_set : list[float]
            A list of Mean Squared Errors (MSE) recorded for the first evaluation set
            at each iteration so far.
        patience : int or None, optional
            The number of iterations to wait for an improvement before stopping.
            If None, this condition for early stopping is disabled.
        target_error : float or None, optional
            A specific MSE value. If the evaluation MSE reaches this value or lower,
            training stops. If None, this condition is disabled.
        tol : float, default=1e-4
            Minimum improvement required to consider as "improvement".
            If MSE decreases by less than tol, it's not considered an improvement.

        Returns
        -------
        bool
            True if the conditions for early stopping are met, False otherwise.
            Returns False if `patience` is None and `target_error` is None, or if
            insufficient iterations have passed to evaluate the patience condition.
        """

        if target_error is not None:
            # If a target error is specified, check if the last MSE is less than or equal to the target error
            if mse_eval_set and mse_eval_set[-1] <= target_error:
                return True
            else:
                return False

        if patience is None:
            return False

        if len(mse_eval_set) < patience:
            return False

        # Check if improvement over `patience` iterations is below tolerance
        # Compare current MSE to MSE from `patience` iterations ago
        old_mse = mse_eval_set[-patience]
        current_mse = mse_eval_set[-1]
        improvement = old_mse - current_mse

        # Stop if no significant improvement
        return improvement < tol

    def _expand_eval_ebm_dataframe_with_best_path(self, best_path, anchor_node_label_name, eval_set=None):
        # we expand the ebm dataframe ONLY by adding the new columns related to the best path, we are not exploring new paths
        if eval_set is not None:
            columns_names = ExtendedBoostingMatrix.get_columns_related_to_path(best_path,
                                                                               self.train_ebm_dataframe_.columns)
            for eval_set_number, eval_set_tuple in enumerate(eval_set):
                if eval_set_tuple is None:
                    continue
                eval_set_dataset, y_eval_set = eval_set_tuple
                # find the new columns in the eval set
                missing_columns = [col for col in columns_names if
                                   col not in self.eval_set_ebm_df_and_target_[eval_set_number][0].columns]
                new_columns_for_eval_set = ExtendedBoostingMatrix.generate_new_columns_from_columns_names(
                    dataset=eval_set_dataset,
                    ebm_to_be_expanded=self.eval_set_ebm_df_and_target_[eval_set_number][0],
                    columns_names=missing_columns,
                    main_label_name=anchor_node_label_name,
                    replace_nan_with=self.replace_nan_with)
                self.eval_set_ebm_df_and_target_[eval_set_number][0] = pd.concat(
                    [self.eval_set_ebm_df_and_target_[eval_set_number][0], new_columns_for_eval_set], axis=1)

    def generate_ebm_for_dataset(self, dataset: list[nx.Graph], columns_names=None):
        """
           Generates an Extended Boosting Matrix (EBM) for a given dataset of graphs.

           The EBM is a pandas DataFrame where rows correspond to graphs and columns
           correspond to features derived from paths in the graphs. If `columns_names`
           is provided, the EBM will only contain these columns. Otherwise, it will
           include columns related to all paths selected during the fitting process
           (stored in `self.paths_selected_by_epb_` and `self.columns_names_`).

           Parameters
           ----------
           dataset : list[nx.Graph]
               A list of NetworkX graph objects for which to generate the EBM.
           columns_names : list[str], optional
               A list of column names to include in the generated EBM. If None,
               columns are determined by the paths selected during fitting.
               Defaults to None.

           Returns
           -------
           pd.DataFrame
               The generated Extended Boosting Matrix.

           Raises
           ------
           AssertionError
               If the model has not been fitted yet (i.e., `self.is_fitted_` is False).
           """

        assert self.is_fitted_
        if columns_names is None:

            selected_path = list(self.paths_selected_by_epb_)
            columns_names = []
            for path in selected_path:
                columns_names += ExtendedBoostingMatrix.get_columns_related_to_path(path=path,
                                                                                    columns_names=self.columns_names_)
        columns_names = list(set(columns_names))
        ebm_dataframe = ExtendedBoostingMatrix.generate_new_columns_from_columns_names(dataset=dataset,
                                                                                       columns_names=columns_names,
                                                                                       main_label_name=self.name_of_label_attribute_,
                                                                                       replace_nan_with=self.replace_nan_with)

        return ebm_dataframe

    def predict(self, X: list[nx.Graph] | nx.Graph | None = None, ebm_dataframe: pd.DataFrame | None = None) -> list[
        numbers.Number]:
        """
        Predicts target values for the given input data.

        The method can accept either a list of NetworkX graphs (`X`) or a pre-computed
        Extended Boosting Matrix (`ebm_dataframe`).

        Parameters
        ----------
        X : list[nx.Graph] | None, default=None
            A list of NetworkX graph objects for which to make predictions.
            Required if `ebm_dataframe` is not provided.
        ebm_dataframe : pd.DataFrame | None, default=None
            A pre-computed Extended Boosting Matrix. If provided, this matrix will be
            used directly for prediction, bypassing the EBM generation from `X`.
            Required if `X` is not provided.

        Returns
        -------
        list[numbers.Number]
            A list of predicted numerical values for the input samples.

        Raises
        ------
        AssertionError
            If the model has not been fitted yet (i.e., `fit` has not been called).
            If neither `X` nor `ebm_dataframe` is provided.
        """
        assert X is not None or ebm_dataframe is not None
        assert self.is_fitted_
        if ebm_dataframe is None:
            if isinstance(X, nx.Graph):
                X = [X]
            ebm_dataframe = self.generate_ebm_for_dataset(dataset=X)
        return self.base_learner_.predict(ebm_dataframe)

    def predict_step_by_step(self, X: list[nx.Graph] | nx.Graph | None = None,
                             ebm_dataframe: pd.DataFrame | None = None) -> list[
        np.array]:
        """
        Generates predictions for each input sample at each boosting iteration.

        This method takes either a list of NetworkX graphs or a precomputed
        Extended Boosting Matrix (EBM) as input. It uses the trained base learner
        to make predictions iteratively, returning a list where each element
        is an array of predictions for all samples at a specific boosting step.

        Parameters
        ----------
        X : list[nx.Graph] | None, default=None
            A list of NetworkX graph objects for which to generate predictions.
            Either `X` or `ebm_dataframe` must be provided.
        ebm_dataframe : pd.DataFrame | None, default=None
            A precomputed Extended Boosting Matrix. If provided, `X` is ignored.
            Either `X` or `ebm_dataframe` must be provided.

        Returns
        -------
        list[np.array]
            A list of NumPy arrays. Each array contains the predictions for all
            input samples at a specific boosting iteration. The outer list
            corresponds to the iterations, and the inner arrays contain
            the predictions.

        Raises
        ------
        AssertionError
            If the model has not been fitted (i.e., `self.is_fitted_` is False).
        AssertionError
            If both `X` and `ebm_dataframe` are None.
        """
        assert X is not None or ebm_dataframe is not None
        assert self.is_fitted_
        if ebm_dataframe is None:
            if isinstance(X, nx.Graph):
                X = [X]
            ebm_dataframe = self.generate_ebm_for_dataset(dataset=X)
        return self.base_learner_.predict_step_by_step(ebm_dataframe)

    def evaluate(self, X: list[nx.Graph] | nx.Graph | None = None, y=None, ebm_dataframe: pd.DataFrame | None = None):
        """
        Evaluates the model on the given dataset and returns the Mean Squared Error (MSE) for each iteration.

        Parameters
        ----------
        X : list[nx.Graph] | None, default=None
            A list of NetworkX graph objects representing the input samples.
        y : array-like
            The true target values corresponding to `X` or `ebm_dataframe`.
        ebm_dataframe : pd.DataFrame | None, default=None
            A pre-generated Extended Boosting Matrix for the input samples.
            If provided, `X` is ignored for EBM generation.

        Returns
        -------
        list[float]
            A list of float values, where each value is the Mean Squared Error
            of the model on the provided dataset at a specific boosting iteration.
            The length of the list corresponds to the number of boosting iterations (`n_iter`).

        Raises
        ------
        AssertionError
            If `y` is None.
            If both `X` and `ebm_dataframe` are None.
            If the model has not been fitted yet (i.e., `fit` has not been called).
        """
        # it returns the evolution of the mse for each iteration
        assert y is not None
        assert X is not None or ebm_dataframe is not None
        assert self.is_fitted_
        if ebm_dataframe is None:
            if isinstance(X, nx.Graph):
                X = [X]
            ebm_dataframe = self.generate_ebm_for_dataset(dataset=X)
        return self.base_learner_.evaluate(ebm_dataframe, y)

    def _expand_ebm_dataframe(self, X: list[nx.Graph], selected_path, main_label_name: str):
        self._ebm_has_been_expanded_in_this_iteration = True
        if selected_path in self.paths_selected_by_epb_:
            return
        elif len(selected_path) >= self.max_path_length:
            self.paths_selected_by_epb_.add(selected_path)
        else:
            self.paths_selected_by_epb_.add(selected_path)
            new_columns = ExtendedBoostingMatrix.new_columns_to_expand_ebm_dataframe_with_path(dataset=X,
                                                                                               selected_path=selected_path,
                                                                                               main_label_name=main_label_name,
                                                                                               df_to_be_expanded=self.train_ebm_dataframe_,
                                                                                               replace_nan_with=self.replace_nan_with)
            self.train_ebm_dataframe_ = pd.concat([self.train_ebm_dataframe_, new_columns], axis=1)

    def _initialize_path_boosting(self, X, list_anchor_nodes_labels: list, main_label_name: str,
                                  eval_set: list[tuple[list[nx.Graph], Iterable]] = None):

        self.name_of_label_attribute = main_label_name

        # greate extended boosting matrix for train dataset
        self.train_ebm_dataframe_ = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
            dataset=X,
            list_anchor_nodes_labels=list_anchor_nodes_labels,
            id_label_name=main_label_name,
            replace_nan_with=self.replace_nan_with)
        self.eval_set_ebm_df_and_target_ = []

        # generate extended boosting matrix for eval dataset
        if eval_set is None:
            pass
        else:
            for eval_tuple in eval_set:
                if eval_tuple is None:
                    self.eval_set_ebm_df_and_target_.append(None)
                    continue
                else:
                    eval_dataset, y_eval_set = eval_tuple
                    # prepare extended boosting matrix for eval dataset
                    eval_set_ebm_dataframe = ExtendedBoostingMatrix.initialize_boosting_matrix_with_anchor_nodes_attributes(
                        dataset=eval_dataset,
                        list_anchor_nodes_labels=list_anchor_nodes_labels,
                        id_label_name=main_label_name,
                        replace_nan_with=self.replace_nan_with)
                    self.eval_set_ebm_df_and_target_.append([eval_set_ebm_dataframe, y_eval_set])

        # initialize base learner wrapper
        self.base_learner_: AdditiveModelWrapper = AdditiveModelWrapper(BaseModelClass=self.BaseLearnerClass,
                                                                        base_model_class_kwargs=self.kwargs_for_base_learner,
                                                                        learning_rate=self.learning_rate,
                                                                        learning_rate_scheduler=self.learning_rate_scheduler)

    @staticmethod
    def _find_best_path(train_ebm_dataframe: pd.DataFrame, y, SelectorClass, kwargs_for_selector) -> tuple[int]:
        """
         Selects the path with the highest importance from a frequency-focused dataframe by training a feature selector,
         identifying the most significant column, and extracting the corresponding path.

         Note:important that this stays as static method because it is used also by the variable importance class, to select variable importance by comparison

         Parameters:
             train_ebm_dataframe (pd.DataFrame): Extended boosting matrix containing path frequency details.
             y (array-like): The target values or negative gradient for path selection.
             SelectorClass: A feature selector (e.g., a regressor) used to determine column importance.
             kwargs_for_selector (dict): Configuration parameters for SelectorClass.

         Returns:
             tuple[int]: The path corresponding to the most important column.
         """

        base_feature_selector = SelectorClass(**kwargs_for_selector)
        frequency_boosting_matrix = ExtendedBoostingMatrix.get_frequency_boosting_matrix(train_ebm_dataframe)

        base_feature_selector = base_feature_selector.fit(X=frequency_boosting_matrix, y=y)
        best_feature_index = np.array(base_feature_selector.feature_importances_).argmax()
        best_feature = frequency_boosting_matrix.columns[best_feature_index]
        best_path = ExtendedBoostingMatrix.get_path_from_column_name(best_feature)

        return best_path

    def _validate_data(
            self,
            X: list[nx.Graph] = "no_validation",
            y="no_validation",
            **check_params,
    ):

        util_validate_data(model=self, X=X, y=y, **check_params)

    def plot_training_and_eval_errors(self, skip_first_n_iterations=False, show=True, save=False,
                                      save_path: str | None = None):
        """
        Plots the training and evaluation set errors (Mean Squared Error) over iterations.

        This method visualizes the progression of the training error and, if
        evaluation sets were provided during fitting, their respective errors
        across the boosting iterations.

        Parameters
        ----------
        skip_first_n_iterations : int or bool, default=False
            If True, a default number of initial iterations (calculated based on
            learning rate) are skipped in the plot, as early iterations can sometimes
            be outliers.
            If an integer, that specific number of initial iterations' errors are skipped.
            If False or 0, all iterations' errors are plotted.
            The actual skipping logic is handled by the underlying
            `plot_training_and_eval_errors` utility function.
        show : bool, default=True
            If True, the plot is displayed.
        save : bool, default=False
            If True, the plot is saved to a file.
        save_path : str | None, default=None
            The directory where the plot will be saved. If None, the current
            working directory is used.

        """
        if hasattr(self, 'fitted_'):
            if not self.fitted_:
                raise ValueError("The model has not been fitted yet. Please call fit() before plotting.")

        if hasattr(self, 'mse_eval_set_'):
            eval_sets_mse = self.mse_eval_set_
        else:
            eval_sets_mse = None
        plot_training_and_eval_errors(learning_rate=self.learning_rate, train_mse=self.train_mse_,
                                      mse_eval_set=eval_sets_mse, skip_first_n_iterations=skip_first_n_iterations,
                                      show=show, save=save, save_path=save_path)

    def plot_variable_importance(self, top_n_features: int | None = None, show: bool = True):
        """
        Plots the computed variable importance scores.

        This method visualizes the importance of features (paths) as determined
        by the SequentialPathBoost model. It uses the `variable_importance_`
        attribute, which is populated during the `fit` method if
        `parameters_variable_importance` was provided at initialization.
        The visual characteristics of the plot are guided by the settings
        contained within `self.parameters_variable_importance`.
        show : bool, default=True
            If True, the plot is displayed.
        """
        if hasattr(self, 'fitted_'):
            if not self.fitted_:
                raise ValueError("The model has not been fitted yet. Please call fit() before plotting.")

        if self.parameters_variable_importance is None:
            raise ValueError(
                "Variable importance is not computed. Please set parameters_variable_importance in the constructor.")
        plot_variable_importance_utils(variable_importance=self.variable_importance_,
                                       parameters_variable_importance=self.parameters_variable_importance,
                                       top_n=top_n_features, show=show)

    def get_mse_for_patience(self, patience: int, eval_set_index: int = 0) -> float:
        """
        Returns the Mean Squared Error (MSE) that we would obtain if we stopped training at the specified patience.
        By default the mse returned is the MSE relative to the first eval_set,
        """
        if not hasattr(self, 'fitted_'):
            raise ValueError("The model has not been fitted yet. Please call fit() before getting MSE for patience.")

        if not hasattr(self, "eval_sets_mse_"):
            raise ValueError(
                "The model has not been evaluated on any evaluation set. Please provide an eval_set during fitting.")

        if len(self.eval_sets_mse_) <= eval_set_index:
            raise ValueError(
                f"Eval set index {eval_set_index} is out of bounds for the number of evaluation sets: {len(self.eval_sets_mse_)}.")
        if len(self.eval_sets_mse_[eval_set_index]) < patience:
            raise ValueError(f"Patience {patience} exceeds the number of training iterations.")

        consecutive_increases = 0
        last_mse_value = self.eval_sets_mse_[eval_set_index][0]
        for error in self.eval_sets_mse_[eval_set_index]:
            if error >= last_mse_value:
                consecutive_increases += 1
            else:
                consecutive_increases = 0
                last_mse_value = error
            if consecutive_increases >= patience:
                return last_mse_value

        # If we never hit the patience condition, return the last MSE value
        return self.eval_sets_mse_[eval_set_index][-1]

    def get_final_eval_set_mse(self):
        """
        Returns the evaluation set MSE if it was computed during fitting.
        """
        if hasattr(self, 'mse_eval_set_'):
            final_eval_set_mse= []
            for mse in self.mse_eval_set_:
                final_eval_set_mse.append(mse[-1])
            return final_eval_set_mse
        else:
            raise AttributeError("Evaluation set MSE is not available. Please fit the model with eval_set.")
