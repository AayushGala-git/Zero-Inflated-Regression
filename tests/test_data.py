"""
Tests for the data module.
"""

import os
import pytest
import numpy as np
import pandas as pd
from data import load_data, split_data, get_feature_target_arrays, load_and_split_data


def test_load_data(sample_data_file):
    """Test loading data from CSV."""
    df = load_data(sample_data_file)
    
    # Check that DataFrame was loaded correctly
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    
    # Check required columns
    required_cols = ['X1', 'X2', 'X3', 'y', 'zero']
    for col in required_cols:
        assert col in df.columns


def test_load_data_subsample(sample_data_file):
    """Test loading data with subsampling."""
    subsample_size = 500
    df = load_data(sample_data_file, subsample_size=subsample_size)
    
    # Check that subsampling worked
    assert len(df) == subsample_size


def test_load_data_nonexistent():
    """Test loading nonexistent data file."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_data.csv")


def test_split_data(sample_data_file):
    """Test splitting data into train and test sets."""
    df = load_data(sample_data_file)
    test_size = 0.2
    
    train_df, test_df = split_data(df, test_size=test_size)
    
    # Check split sizes
    assert len(train_df) + len(test_df) == len(df)
    assert abs(len(test_df) / len(df) - test_size) < 0.01  # Allow small rounding differences
    
    # Check that stratification worked (similar zero rate)
    zero_rate_orig = df['zero'].mean()
    zero_rate_train = train_df['zero'].mean()
    zero_rate_test = test_df['zero'].mean()
    
    assert abs(zero_rate_train - zero_rate_orig) < 0.05
    assert abs(zero_rate_test - zero_rate_orig) < 0.05


def test_get_feature_target_arrays(sample_data_file):
    """Test extracting feature and target arrays."""
    df = load_data(sample_data_file)
    
    X, y_zero, y_count = get_feature_target_arrays(df)
    
    # Check shapes
    assert X.shape[0] == len(df)
    assert X.shape[1] == 3  # Default 3 features
    assert y_zero.shape[0] == len(df)
    assert y_count.shape[0] == len(df)
    
    # Check feature types
    assert X.dtype == np.float32
    
    # Check with custom feature columns
    feature_cols = ['X1', 'X3']
    X_custom, _, _ = get_feature_target_arrays(df, feature_cols=feature_cols)
    assert X_custom.shape[1] == 2


def test_load_and_split_data(sample_config, sample_data_file, monkeypatch):
    """Test loading and splitting data in one step."""
    # Patch the config to use the sample data file
    monkeypatch.setattr(sample_config.data, "input_file", sample_data_file)
    
    data_dict = load_and_split_data(sample_config)
    
    # Check that all expected components are present
    expected_keys = [
        'train_df', 'test_df', 
        'X_train', 'y_zero_train', 'y_count_train',
        'X_test', 'y_zero_test', 'y_count_test'
    ]
    
    for key in expected_keys:
        assert key in data_dict
    
    # Check dimensions
    assert len(data_dict['train_df']) > 0
    assert len(data_dict['test_df']) > 0
    assert data_dict['X_train'].shape[0] == len(data_dict['train_df'])
    assert data_dict['X_test'].shape[0] == len(data_dict['test_df'])
    
    # Check that train and test are different
    assert not np.array_equal(data_dict['X_train'], data_dict['X_test'])
    
    # Check that we can run with subsample
    monkeypatch.setattr(sample_config.data, "subsample_size", 500)
    data_dict_sub = load_and_split_data(sample_config)
    assert len(data_dict_sub['train_df']) + len(data_dict_sub['test_df']) == 500
