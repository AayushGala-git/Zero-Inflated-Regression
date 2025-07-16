"""
Tests for the features module.
"""

import pytest
import numpy as np
import pandas as pd
from features import (
    RiskScoreTransformer, 
    InteractionTransformer, 
    create_feature_pipeline, 
    transform_features
)


def test_risk_score_transformer():
    """Test the RiskScoreTransformer class."""
    # Create sample data
    np.random.seed(42)
    X = np.random.rand(100, 3).astype(np.float32)
    y = (np.random.rand(100) > 0.7).astype(int)  # 30% ones
    
    # Initialize and fit transformer
    transformer = RiskScoreTransformer(random_state=42)
    transformer.fit(X, y)
    
    # Check that weights were learned
    assert transformer.weights_ is not None
    assert transformer.weights_.shape == (3,)
    
    # Check transformation
    X_transformed = transformer.transform(X)
    assert X_transformed.shape == (100, 4)  # Original features + risk score
    
    # Check that risk score is different for different classes
    risk_scores = X_transformed[:, -1]
    avg_risk_pos = risk_scores[y == 1].mean()
    avg_risk_neg = risk_scores[y == 0].mean()
    assert avg_risk_pos != pytest.approx(avg_risk_neg)
    
    # Check that unfitted transformer raises error
    unfitted = RiskScoreTransformer()
    with pytest.raises(ValueError):
        unfitted.transform(X)


def test_interaction_transformer():
    """Test the InteractionTransformer class."""
    # Create sample data
    np.random.seed(42)
    X = np.random.rand(100, 3).astype(np.float32)
    
    # Initialize transformer with default interaction (X1 × X3)
    transformer = InteractionTransformer(interactions=[(0, 2)])
    transformer.fit(X)  # No-op
    
    # Check transformation
    X_transformed = transformer.transform(X)
    assert X_transformed.shape == (100, 4)  # Original features + 1 interaction
    
    # Verify interaction term is correct
    expected_interaction = X[:, 0] * X[:, 2]
    np.testing.assert_array_almost_equal(X_transformed[:, 3], expected_interaction)
    
    # Test with multiple interactions
    transformer = InteractionTransformer(interactions=[(0, 1), (1, 2)])
    X_transformed = transformer.transform(X)
    assert X_transformed.shape == (100, 5)  # Original features + 2 interactions
    
    # Test with invalid interaction indices
    transformer = InteractionTransformer(interactions=[(0, 10)])
    with pytest.raises(ValueError):
        transformer.transform(X)


def test_create_feature_pipeline(sample_config):
    """Test creating feature pipeline from config."""
    pipeline_info = create_feature_pipeline(sample_config)
    
    # Check that pipeline and feature names are returned
    assert 'pipeline' in pipeline_info
    assert 'feature_names' in pipeline_info
    
    # Check pipeline steps
    pipeline = pipeline_info['pipeline']
    step_names = [name for name, _ in pipeline.steps]
    
    # Should include spline, risk_score, interaction, and scaler steps
    assert 'spline' in step_names
    assert 'risk_score' in step_names
    assert 'interaction' in step_names
    assert 'scaler' in step_names
    
    # Check that feature names are updated
    feature_names = pipeline_info['feature_names']
    assert len(feature_names) > 3  # Should have more features than original


def test_transform_features(sample_config):
    """Test transforming features with the pipeline."""
    # Create sample data
    np.random.seed(42)
    X_train = np.random.rand(100, 3).astype(np.float32)
    X_test = np.random.rand(50, 3).astype(np.float32)
    y_zero_train = (np.random.rand(100) > 0.7).astype(int)
    
    # Transform features
    feature_dict = transform_features(
        X_train=X_train,
        X_test=X_test,
        y_zero_train=y_zero_train,
        config=sample_config,
        verbose=True
    )
    
    # Check that transformed data is returned
    assert 'X_train_transformed' in feature_dict
    assert 'X_test_transformed' in feature_dict
    assert 'pipeline' in feature_dict
    assert 'feature_names' in feature_dict
    
    # Check shapes
    X_train_transformed = feature_dict['X_train_transformed']
    X_test_transformed = feature_dict['X_test_transformed']
    
    assert X_train_transformed.shape[0] == 100
    assert X_test_transformed.shape[0] == 50
    assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
    assert X_train_transformed.shape[1] > 3  # More features after transformation
    
    # Check that training and test sets are different
    assert not np.array_equal(
        X_train_transformed[:1], 
        X_test_transformed[:1]
    )
