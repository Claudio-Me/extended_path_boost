import pandas as pd
import pytest
from extended_path_boost.utils.classes.extended_boosting_matrix import ExtendedBoostingMatrix


def test_get_frequency_boosting_matrix_single_column():
    df = pd.DataFrame({
        'n_times_present_feature1': [1, 2, 3],
        'unrelated_column': [4, 5, 6]
    })
    result = ExtendedBoostingMatrix.get_frequency_boosting_matrix(df)
    assert 'n_times_present_feature1' in result.columns
    assert 'unrelated_column' not in result.columns
    assert len(result.columns) == 1
    assert result['n_times_present_feature1'].tolist() == [1, 2, 3]


def test_get_frequency_boosting_matrix_multiple_columns():
    df = pd.DataFrame({
        'n_times_present_feature1': [1, 2, 3],
        'n_times_present_feature2': [4, 5, 6],
        'unrelated_column': [7, 8, 9]
    })
    result = ExtendedBoostingMatrix.get_frequency_boosting_matrix(df)
    assert 'n_times_present_feature1' in result.columns
    assert 'n_times_present_feature2' in result.columns
    assert 'unrelated_column' not in result.columns
    assert len(result.columns) == 2
    assert result['n_times_present_feature1'].tolist() == [1, 2, 3]
    assert result['n_times_present_feature2'].tolist() == [4, 5, 6]


def test_get_frequency_boosting_matrix_no_matching_columns():
    df = pd.DataFrame({
        'unrelated_column1': [1, 2, 3],
        'unrelated_column2': [4, 5, 6]
    })
    result = ExtendedBoostingMatrix.get_frequency_boosting_matrix(df)
    assert result.empty
    assert len(result.columns) == 0


def test_get_frequency_boosting_matrix_empty_dataframe():
    df = pd.DataFrame()
    result = ExtendedBoostingMatrix.get_frequency_boosting_matrix(df)
    assert result.empty
    assert len(result.columns) == 0


def test_get_frequency_boosting_matrix_with_index():
    df = pd.DataFrame({
        'n_times_present_feature1': [1, 2, 3],
        'n_times_present_feature2': [4, 5, 6]
    }, index=['a', 'b', 'c'])
    result = ExtendedBoostingMatrix.get_frequency_boosting_matrix(df)
    assert list(result.index) == ['a', 'b', 'c']
    assert 'n_times_present_feature1' in result.columns
    assert 'n_times_present_feature2' in result.columns
    assert result['n_times_present_feature1'].tolist() == [1, 2, 3]
    assert result['n_times_present_feature2'].tolist() == [4, 5, 6]
