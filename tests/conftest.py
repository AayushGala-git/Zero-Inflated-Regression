"""
Test configuration for the zero-inflated count model pipeline.
"""

import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from config import Config, load_config


@pytest.fixture
def sample_config():
    """Create a minimal sample configuration for testing."""
    config_dict = {
        "general": {
            "random_seed": 42,
            "n_jobs": -1,
            "log_level": "INFO"
        },
        "data": {
            "input_file": "test_data.csv",
            "test_size": 0.2,
            "subsample_size": None
        },
        "features": {
            "spline": {
                "degree": 3,
                "n_knots": 4,
                "include_bias": False
            },
            "generate_risk_score": True,
            "generate_interaction": True
        },
        "feature_pruning": {
            "enabled": True,
            "shap_sample_size": 100,
            "top_k": 5
        },
        "zero_stage": {
            "model_types": ["xgb"],
            "stacking": False,
            "hyperparameters": {
                "xgb": {
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "subsample": 0.8,
                    "n_estimators": 10
                }
            }
        },
        "count_stage": {
            "model_types": ["xgb"],
            "stacking": False,
            "hyperparameters": {
                "xgb": {
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "subsample": 0.8,
                    "n_estimators": 10
                }
            },
            "sample_weight_factor": 1.0
        },
        "tail_modeling": {
            "enabled": False,
            "threshold_percentile": 95,
            "models": ["xgb"]
        },
        "optuna": {
            "enabled": False,
            "n_trials": 5,
            "timeout": None,
            "study_direction": "maximize",
            "pruner": "median"
        },
        "output": {
            "directory": "test_output",
            "save_models": True,
            "save_plots": True,
            "plot_formats": ["png"],
            "dpi": 300
        },
        "mlflow": {
            "enabled": False,
            "tracking_uri": None,
            "experiment_name": "test_experiment"
        }
    }
    return Config(**config_dict)


@pytest.fixture
def sample_data_file():
    """Create a small sample CSV file for testing."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "test_data.csv")
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 1000
        
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.uniform(-1, 1, n_samples)
        X3 = np.random.binomial(1, 0.3, n_samples)
        
        # Generate zero-inflation probabilities
        logit_p = -0.5 + 1.0 * X1 - 0.8 * X2 + 0.5 * X3
        pi = 1 / (1 + np.exp(-logit_p))
        
        # Generate count means
        log_mu = 1.0 + 0.7 * X1 + 0.3 * X2 - 0.2 * X3
        mu = np.exp(log_mu)
        
        # Generate counts from Poisson
        from scipy.stats import poisson
        counts = poisson.rvs(mu, size=n_samples)
        
        # Apply zero-inflation
        is_zero = np.random.rand(n_samples) < pi
        y = counts.copy()
        y[is_zero] = 0
        
        # Create DataFrame
        df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'X3': X3,
            'y': y,
            'zero': (y == 0).astype(int)
        })
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        
        yield file_path


@pytest.fixture
def config_file(sample_config):
    """Create a temporary config file for testing."""
    import yaml
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, "test_config.yaml")
        
        # Convert Config to dict
        config_dict = sample_config.dict()
        
        # Write to YAML file
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        yield config_path
