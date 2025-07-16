"""
Tests for the configuration module.
"""

import os
import pytest
from pathlib import Path
from config import load_config, Config

def test_load_config_valid(config_file):
    """Test loading a valid configuration file."""
    config = load_config(config_file)
    assert isinstance(config, Config)
    assert config.general.random_seed == 42
    assert config.data.test_size == 0.2
    assert config.zero_stage.model_types == ["xgb"]


def test_load_config_nonexistent():
    """Test loading a nonexistent configuration file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")


def test_config_validation():
    """Test configuration validation."""
    # Test with invalid test_size
    with pytest.raises(ValueError):
        Config(
            general={"random_seed": 42, "n_jobs": -1, "log_level": "INFO"},
            data={"input_file": "data.csv", "test_size": 1.5},  # Invalid test_size
            features={"spline": {"degree": 3, "n_knots": 4, "include_bias": False}},
            feature_pruning={"enabled": True, "shap_sample_size": 100, "top_k": 5},
            zero_stage={
                "model_types": ["xgb"],
                "hyperparameters": {"xgb": {"learning_rate": 0.1, "max_depth": 3}}
            },
            count_stage={
                "model_types": ["xgb"],
                "hyperparameters": {"xgb": {"learning_rate": 0.1, "max_depth": 3}}
            },
            tail_modeling={"enabled": False, "threshold_percentile": 95},
            optuna={"enabled": False, "n_trials": 5},
            output={"directory": "output", "save_models": True},
            mlflow={"enabled": False, "experiment_name": "test"}
        )

    # Test with missing hyperparameters for model type
    with pytest.raises(ValueError):
        Config(
            general={"random_seed": 42, "n_jobs": -1, "log_level": "INFO"},
            data={"input_file": "data.csv", "test_size": 0.2},
            features={"spline": {"degree": 3, "n_knots": 4, "include_bias": False}},
            feature_pruning={"enabled": True, "shap_sample_size": 100, "top_k": 5},
            zero_stage={
                "model_types": ["xgb", "lgb"],  # Two model types
                "hyperparameters": {"xgb": {"learning_rate": 0.1}}  # Missing lgb hyperparameters
            },
            count_stage={
                "model_types": ["xgb"],
                "hyperparameters": {"xgb": {"learning_rate": 0.1, "max_depth": 3}}
            },
            tail_modeling={"enabled": False, "threshold_percentile": 95},
            optuna={"enabled": False, "n_trials": 5},
            output={"directory": "output", "save_models": True},
            mlflow={"enabled": False, "experiment_name": "test"}
        )


def test_config_defaults(sample_config):
    """Test that configuration defaults are set correctly."""
    assert sample_config.general.log_level == "INFO"
    assert sample_config.data.test_size == 0.2
    assert sample_config.features.spline.degree == 3
    assert sample_config.feature_pruning.top_k == 5
    assert sample_config.count_stage.sample_weight_factor == 1.0
    assert sample_config.output.dpi == 300
