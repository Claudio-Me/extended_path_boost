import numbers
import networkx as nx
import numpy as np
from typing import Iterable
from .classes.interfaces.interface_base_learner import BaseLearnerClassInterface
from .classes.interfaces.interface_selector import SelectorClassInterface
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import validate_data
import inspect

def check_interface(class_to_be_checked, interface_class):
    if not issubclass(class_to_be_checked, interface_class):
        missing_methods = []
        interface_methods = inspect.getmembers(interface_class, predicate=inspect.isfunction)
        class_methods = inspect.getmembers(class_to_be_checked, predicate=inspect.isfunction)
        interface_method_names = [name for name, _ in interface_methods]
        class_method_names = [name for name, _ in class_methods]

        for method in interface_method_names:
            if method not in class_method_names:
                missing_methods.append(method)

        missing_attributes = []
        interface_attributes = inspect.getmembers(interface_class, predicate=lambda x: not(inspect.isroutine(x)))
        class_attributes = inspect.getmembers(class_to_be_checked, predicate=lambda x: not(inspect.isroutine(x)))
        interface_attribute_names = [name for name, _ in interface_attributes]
        class_attribute_names = [name for name, _ in class_attributes]

        for attribute in interface_attribute_names:
            if attribute not in class_attribute_names:
                missing_attributes.append(attribute)

        missing_items = missing_methods + missing_attributes
        raise TypeError(f"{class_to_be_checked.__name__} must implement {interface_class.__name__}. Missing items: {missing_items}")




def util_validate_data(
        model,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        **check_params,
):
    # We use the `_validate_data` method to validate the input data.
    # This method is defined in the `BaseEstimator` class.
    # It allows to:
    # - run different checks on the input data
    if not np.array_equal(X, "no_validation"):
        assert isinstance(X, list) and all(isinstance(item, nx.Graph) for item in X)
    if not np.array_equal(y, "no_validation"):
        assert isinstance(y, Iterable) and all(isinstance(item, numbers.Number) for item in y)

    # check BaseLearnerClass and SelectorClass respects the respective interfaces
    check_interface(model.BaseLearnerClass, BaseLearnerClassInterface)

    check_interface(model.SelectorClass, SelectorClassInterface)


    # ------------------------------------------------------------------------------------------------------
    # the following is just to set the default parameters for the selector class ant the base learner class
    if issubclass(model.BaseLearnerClass, DecisionTreeRegressor):
        if model.kwargs_for_base_learner is None:
            model.kwargs_for_base_learner = model._default_kwargs_for_base_learner
        else:
            for key in model._default_kwargs_for_base_learner:
                if key not in model.kwargs_for_base_learner:
                    model.kwargs_for_base_learner[key] = model._default_kwargs_for_base_learner[key]

    if issubclass(model.SelectorClass, DecisionTreeRegressor):
        if model.kwargs_for_selector is None:
            model.kwargs_for_selector = model._default_kwargs_for_selector
        else:
            for key in model._default_kwargs_for_selector:
                if key not in model.kwargs_for_selector:
                    model.kwargs_for_selector[key] = model._default_kwargs_for_selector[key]

    # ------------------------------------------------------------------------------------------------------

    if model.kwargs_for_selector is None:
        model.kwargs_for_selector = {}
    if model.kwargs_for_base_learner is None:
        model.kwargs_for_base_learner = {}

    # check parameters for variable importance
    parameters_variable_importance = check_params.get('parameters_variable_importance', None)
    if parameters_variable_importance is not None:
        assert isinstance(parameters_variable_importance, dict)
        for key, value in parameters_variable_importance.items():
            assert isinstance(key, str)
            if key == 'criterion':
                assert value in ['absolute', 'relative']
            elif key == 'error_used':
                assert value in ['mse', 'mae']
            elif key == 'use_correlation':
                assert isinstance(value, bool)
            elif key == 'normalize':
                assert isinstance(value, bool)
            elif key == 'normalization_value':
                assert value is None or isinstance(value, float)

            else:
                raise ValueError(f"Unknown parameter {key} for variable importance")


    # check list_anchor_nodes_labels
    list_anchor_nodes_labels = check_params.get('list_anchor_nodes_labels', None)
    if list_anchor_nodes_labels is not None:
        # Ensure each element in list_anchor_nodes_labels is a tuple
        len_list_anchor_nodes_labels = len(list_anchor_nodes_labels)
        for i in range(len_list_anchor_nodes_labels):
            if not isinstance(list_anchor_nodes_labels[i], tuple):
                if hasattr(list_anchor_nodes_labels[i], '__iter__') and not isinstance(list_anchor_nodes_labels[i],
                                                                                       str):
                    list_anchor_nodes_labels[i] = tuple(list_anchor_nodes_labels[i])
                else:
                    list_anchor_nodes_labels[i] = tuple([list_anchor_nodes_labels[i]])

    model.list_anchor_nodes_labels = list_anchor_nodes_labels

    # check m_stops
    # only for cyclic path boost
    m_stops = check_params.get('m_stops', None)
    if m_stops is not None:
        assert isinstance(m_stops, list) and all(isinstance(item, (int, type(None))) for item in m_stops)
        assert len(m_stops) == len(model.list_anchor_nodes_labels)


    # check eval sets
    eval_set = check_params.get('eval_set', None)
    if eval_set is not None:
        assert isinstance(eval_set, Iterable) and all(
            isinstance(eval_tuple, tuple) and len(eval_tuple) == 2 for eval_tuple in eval_set)
        for eval_tuple in eval_set:
            assert isinstance(eval_tuple[0], list) and all(isinstance(item, nx.Graph) for item in eval_tuple[0])
            assert isinstance(eval_tuple[1], Iterable) and all(
                isinstance(item, numbers.Number) for item in eval_tuple[1])

    # check patience
    patience = check_params.get('patience', None)
    if patience is not None:
        assert isinstance(patience, int) and patience >= 0, "patience must be a non-negative integer"
        if check_params.get('eval_set', None) is None:
            print("patience is set to None because there is no eval_set provided")
            patience = None
    model.patience = patience


    if not np.array_equal(y, "no_validation"):
        validate_data(model,
                      X="no_validation",
                      y=y,
                      reset=reset,
                      validate_separately=validate_separately
                      )

    if not np.array_equal(X, "no_validation") and not np.array_equal(y, "no_validation"):
        return X, y
    elif not np.array_equal(X, "no_validation"):
        return X
    elif not np.array_equal(y, "no_validation"):
        return y
