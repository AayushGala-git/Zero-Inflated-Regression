"""
Configuration loader for the zero-inflated count model pipeline.

This module provides functions for loading and validating configuration from YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, field_validator, model_validator


class GeneralConfig(BaseModel):
    """General configuration settings."""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    output_dir: str = 'outputs/'


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""
    path: str
    features: List[str]
    target: str


class FeatureEngineeringConfig(BaseModel):
    """Feature engineering configuration."""
    create_polynomial_features: bool = True
    polynomial_degree: int = 2
    create_log_features: bool = True
    create_interaction_features: bool = True
    create_binned_features: bool = True
    n_bins: int = 10
    create_statistical_features: bool = True
    random_state: int = 42


class TrainingConfig(BaseModel):
    """Training configuration."""
    model_type: str
    hyperparameters_path: str
    test_size: float = 0.2
    random_state: int = 42


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    calibration_n_bins: int = 15


class TuningConfig(BaseModel):
    """Configuration for hyperparameter tuning."""
    enabled: bool
    n_trials: int
    zero_hyperparameter_path: str
    count_hyperparameter_path: str


class Config(BaseModel):
    """Main configuration model."""
    general: GeneralConfig
    data: DataConfig
    feature_engineering: FeatureEngineeringConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    tuning: TuningConfig

    class Config:
        # This allows the model to be created from a dictionary
        # that has extra keys, which will be ignored.
        extra = 'ignore'


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from a YAML file and validate it.
    
    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file
        
    Returns
    -------
    Config
        Validated configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)


def get_default_config_path() -> Path:
    """
    Get the default configuration file path.
    
    Returns:
        Path to the default configuration file
    """
    # Look for config.yaml in the current directory and parent directories
    current_dir = Path.cwd()
    potential_paths = [
        current_dir / "config.yaml",
        current_dir.parent / "config.yaml",
        Path(__file__).parent / "config.yaml",
        Path(__file__).parent.parent / "config.yaml",
    ]
    
    for path in potential_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError("Default configuration file (config.yaml) not found")


if __name__ == "__main__":
    # Example usage
    try:
        config = load_config("config.yaml")
        print(f"Configuration loaded successfully:")
        print(f"- Random seed: {config.general.random_seed}")
        print(f"- Input file: {config.data.input_file}")
        print(f"- Model types (zero stage): {config.zero_stage.model_types}")
        print(f"- Feature pruning enabled: {config.feature_pruning.enabled}")
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
