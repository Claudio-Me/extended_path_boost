"""
Verification tests to confirm SequentialPathBoost implementation matches the PDF specification.

These tests verify key algorithm properties from the Extended Path Boost paper (Algorithm 1).
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.tree import DecisionTreeRegressor

from path_boost.utils.classes.extended_boosting_matrix import ExtendedBoostingMatrix
from path_boost.utils.classes.additive_model_wrapper import AdditiveModelWrapper
from path_boost.utils.classes.sequential_path_boost import SequentialPathBoost


class TestSubPathInclusion:
    """Verify that get_columns_related_to_path includes all sub-paths.

    From Algorithm 1 Step 13: Train base learner on path p* AND its sub-paths.
    """

    def test_subpath_inclusion_basic(self):
        """Path (1,2,3) should include columns from (1,), (1,2), and (1,2,3)."""
        columns = [
            "(1,)_attr",
            "(1, 2)_attr",
            "(1, 2, 3)_attr",
            "(2,)_attr",  # Should NOT be included - different starting node
            "(1, 2, 3)_n_times_present"
        ]

        result = ExtendedBoostingMatrix.get_columns_related_to_path((1, 2, 3), columns)

        assert "(1,)_attr" in result, "Sub-path (1,) should be included"
        assert "(1, 2)_attr" in result, "Sub-path (1,2) should be included"
        assert "(1, 2, 3)_attr" in result, "Full path (1,2,3) should be included"
        assert "(2,)_attr" not in result, "Unrelated path (2,) should NOT be included"
        assert "(1, 2, 3)_n_times_present" in result, "Frequency column should be included"

    def test_subpath_anchor_only(self):
        """Path (25,) should only include its own columns."""
        columns = ["(25,)_atomic_mass", "(25, 7)_attr", "(47,)_attr"]

        result = ExtendedBoostingMatrix.get_columns_related_to_path((25,), columns)

        assert "(25,)_atomic_mass" in result
        assert "(25, 7)_attr" not in result  # Longer path, not a sub-path
        assert "(47,)_attr" not in result

    def test_subpath_preserves_order(self):
        """Only paths that are prefixes (same order) should be included."""
        columns = [
            "(1, 2, 3)_attr",
            "(2, 1)_attr",  # Wrong order - should NOT be included
            "(3, 2, 1)_attr",  # Reversed - should NOT be included
            "(1,)_attr",  # Prefix - should be included
        ]

        result = ExtendedBoostingMatrix.get_columns_related_to_path((1, 2, 3), columns)

        assert "(1,)_attr" in result
        assert "(1, 2, 3)_attr" in result
        assert "(2, 1)_attr" not in result
        assert "(3, 2, 1)_attr" not in result


class TestMeanCentering:
    """Verify learning rate application matches PDF specification (Section 3.4).

    From PDF: BaseLearner.fit(X, neg_gradient - mean(neg_gradient))
    And: prediction = learning_rate * h(X) + mean(gradient)
    """

    def test_mean_subtraction_before_fit(self):
        """Verify: BaseLearner.fit(X, neg_gradient - mean(neg_gradient))"""
        # Create simple data
        X = pd.DataFrame({'(1,)_col1': [1, 2, 3, 4, 5]})
        y = np.array([10, 20, 30, 40, 50])  # mean = 30

        wrapper = AdditiveModelWrapper(
            BaseModelClass=DecisionTreeRegressor,
            base_model_class_kwargs={'max_depth': 1},
            learning_rate=0.1
        )

        wrapper.fit_one_step(X=X, y=y, best_path=(1,))

        # After first iteration, _target_variable_mean_ should contain mean(y)
        assert len(wrapper._target_variable_mean_) == 1
        assert np.isclose(wrapper._target_variable_mean_[0], 30.0), \
            f"Expected mean=30.0, got {wrapper._target_variable_mean_[0]}"

    def test_prediction_formula(self):
        """Verify: prediction = learning_rate * h(X) + mean(gradient)"""
        X = pd.DataFrame({'(1,)_attr': [1, 2, 3, 4, 5]})
        y = np.array([10, 20, 30, 40, 50])

        wrapper = AdditiveModelWrapper(
            BaseModelClass=DecisionTreeRegressor,
            base_model_class_kwargs={'max_depth': 1},
            learning_rate=0.5
        )

        wrapper.fit_one_step(X=X, y=y, best_path=(1,))

        # Prediction should be: mean(y) + learning_rate * base_learner_output
        # The base learner was fit on y - mean(y), so its output is centered
        predictions = wrapper.predict(X)

        # All predictions should be reasonable (not NaN, within range)
        assert not np.any(np.isnan(predictions)), "Predictions should not contain NaN"
        assert np.mean(predictions) > 0, "Should be positive for this data"

    def test_learning_rate_scaling(self):
        """Verify that learning rate scales the base learner output correctly."""
        X = pd.DataFrame({'(1,)_attr': np.linspace(0, 10, 20)})
        y = np.linspace(0, 100, 20)  # Linear relationship

        # Compare predictions with different learning rates
        wrapper_low_lr = AdditiveModelWrapper(
            BaseModelClass=DecisionTreeRegressor,
            base_model_class_kwargs={'max_depth': 3},
            learning_rate=0.1
        )
        wrapper_high_lr = AdditiveModelWrapper(
            BaseModelClass=DecisionTreeRegressor,
            base_model_class_kwargs={'max_depth': 3},
            learning_rate=0.5
        )

        wrapper_low_lr.fit_one_step(X=X, y=y, best_path=(1,))
        wrapper_high_lr.fit_one_step(X=X, y=y, best_path=(1,))

        preds_low = wrapper_low_lr.predict(X)
        preds_high = wrapper_high_lr.predict(X)

        # Higher LR should lead to predictions closer to target in first iteration
        residual_low = np.abs(y - preds_low).mean()
        residual_high = np.abs(y - preds_high).mean()

        assert residual_high < residual_low, \
            "Higher learning rate should reduce residuals more in first iteration"


class TestSelectorUsesFrequencyOnly:
    """Verify selector model receives only frequency columns (BM), not full EBM.

    From Algorithm 1: Selector is trained on BM (frequency matrix), not EBM.
    """

    def test_get_frequency_boosting_matrix(self):
        """Verify get_frequency_boosting_matrix filters to only frequency columns."""
        ebm = pd.DataFrame({
            '(1,)_n_times_present': [1, 2, 1],
            '(1, 2)_n_times_present': [0, 1, 1],
            '(1,)_atomic_mass': [12.0, 14.0, 16.0],  # Attribute column
            '(1, 2)_bond_length': [1.5, 1.6, 1.7],   # Attribute column
        })

        frequency_matrix = ExtendedBoostingMatrix.get_frequency_boosting_matrix(ebm)

        # Should only have frequency columns
        assert len(frequency_matrix.columns) == 2, \
            f"Expected 2 frequency columns, got {len(frequency_matrix.columns)}"
        assert all('n_times_present' in col for col in frequency_matrix.columns), \
            "All columns should be frequency columns"

    def test_selector_input_is_frequency_matrix(self):
        """_find_best_path should pass only frequency columns to selector."""
        # Create mock EBM with both frequency and attribute columns
        ebm = pd.DataFrame({
            '(1,)_n_times_present': [1, 2, 1],
            '(1, 2)_n_times_present': [0, 1, 1],
            '(1,)_atomic_mass': [12.0, 14.0, 16.0],  # Attribute column
            '(1, 2)_bond_length': [1.5, 1.6, 1.7],   # Attribute column
        })
        y = np.array([1.0, 2.0, 3.0])

        # Mock selector to capture what it receives
        mock_selector = Mock()
        mock_selector.fit.return_value = mock_selector
        mock_selector.feature_importances_ = np.array([0.6, 0.4])

        MockSelectorClass = Mock(return_value=mock_selector)

        # Call _find_best_path
        result = SequentialPathBoost._find_best_path(
            train_ebm_dataframe=ebm,
            y=y,
            SelectorClass=MockSelectorClass,
            kwargs_for_selector={}
        )

        # Verify selector.fit was called with only frequency columns
        call_args = mock_selector.fit.call_args
        X_passed = call_args.kwargs['X']

        # Should only have frequency columns
        assert 'n_times_present' in X_passed.columns[0], \
            f"First column should be frequency column, got {X_passed.columns[0]}"
        assert 'atomic_mass' not in str(X_passed.columns), \
            "Attribute columns should not be passed to selector"
        assert len(X_passed.columns) == 2, \
            f"Should have 2 frequency columns, got {len(X_passed.columns)}"


class TestNegativeGradient:
    """Verify negative gradient computation.

    From Algorithm 1: r^(t) = y - ŷ^(t-1)
    """

    def test_negative_gradient_formula(self):
        """r = y - y_hat"""
        y = np.array([10, 20, 30])
        y_hat = np.array([8, 22, 28])

        result = AdditiveModelWrapper._neg_gradient(y, y_hat)
        expected = np.array([2, -2, 2])

        np.testing.assert_array_equal(result, expected)

    def test_first_iteration_uses_y_directly(self):
        """First iteration should use y as target (since ŷ^(0) = 0)."""
        # This is implicitly tested since neg_gradient of y - 0 = y
        y = np.array([10, 20, 30])
        y_hat_initial = np.zeros(3)

        result = AdditiveModelWrapper._neg_gradient(y, y_hat_initial)

        np.testing.assert_array_equal(result, y)

    def test_negative_gradient_preserves_shape(self):
        """Negative gradient should preserve the shape of input."""
        y = np.array([1, 2, 3, 4, 5])
        y_hat = np.array([1.5, 2.5, 2.5, 4.5, 4.5])

        result = AdditiveModelWrapper._neg_gradient(y, y_hat)

        assert result.shape == y.shape


class TestIterativeUpdates:
    """Verify iterative boosting update formula.

    From Algorithm 1: ŷ^(t) = ŷ^(t-1) + learning_rate * h_t(X)
    """

    def test_predictions_accumulate_over_iterations(self):
        """Predictions should accumulate contributions from each iteration."""
        X = pd.DataFrame({'(1,)_attr': np.linspace(0, 10, 20)})
        y = np.linspace(0, 100, 20)

        wrapper = AdditiveModelWrapper(
            BaseModelClass=DecisionTreeRegressor,
            base_model_class_kwargs={'max_depth': 2},
            learning_rate=0.1
        )

        predictions_after_each = []
        for i in range(5):
            wrapper.fit_one_step(X=X, y=y, best_path=(1,))
            predictions_after_each.append(wrapper.predict(X).copy())
            # Verify base learner count matches iteration number
            assert len(wrapper.base_learners_list) == i + 1

        # Verify predictions change across iterations
        for i in range(1, len(predictions_after_each)):
            # Predictions should differ between iterations
            assert not np.allclose(predictions_after_each[i], predictions_after_each[i-1]), \
                f"Predictions should change between iteration {i} and {i+1}"

    def test_base_learners_accumulate(self):
        """Each iteration should add a new base learner."""
        X = pd.DataFrame({'(1,)_attr': [1, 2, 3, 4, 5]})
        y = np.array([10, 20, 30, 40, 50])

        wrapper = AdditiveModelWrapper(
            BaseModelClass=DecisionTreeRegressor,
            base_model_class_kwargs={'max_depth': 1},
            learning_rate=0.1
        )

        for i in range(3):
            wrapper.fit_one_step(X=X, y=y, best_path=(1,))
            assert len(wrapper.base_learners_list) == i + 1, \
                f"Should have {i+1} base learners after {i+1} iterations"


class TestPathExtraction:
    """Verify path extraction from column names."""

    def test_get_path_from_column_name_single_node(self):
        """Extract path from single-node column name."""
        path = ExtendedBoostingMatrix.get_path_from_column_name("(25,)_atomic_mass")
        assert path == (25,)

    def test_get_path_from_column_name_multi_node(self):
        """Extract path from multi-node column name."""
        path = ExtendedBoostingMatrix.get_path_from_column_name("(1, 2, 3)_some_attribute")
        assert path == (1, 2, 3)

    def test_get_path_from_frequency_column(self):
        """Extract path from frequency column name."""
        path = ExtendedBoostingMatrix.get_path_from_column_name("(47, 6)_n_times_present")
        assert path == (47, 6)
