"""
Data module for loading and preprocessing zero-inflated count data.

This module handles data loading, preprocessing, and train/test splits for
zero-inflated count modeling.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import Config

# Set up module logger
logger = logging.getLogger(__name__)


def load_data(file_path: Union[str, Path], subsample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from a CSV file with optional subsampling.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file containing the zero-inflated count data
    subsample_size : int, optional
        Number of samples to randomly select if subsampling is needed
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file is empty or does not contain required columns
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Could not find data file: {file_path}")
    
    # Validate dataframe
    if len(df) == 0:
        msg = f"Empty dataframe loaded from {file_path}"
        logger.error(msg)
        raise ValueError(msg)
        
    required_cols = ['X1', 'X2', 'X3', 'y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        msg = f"Missing required columns: {missing_cols}"
        logger.error(msg)
        raise ValueError(msg)
    
    # Generate zero indicator if not present
    if 'zero' not in df.columns:
        logger.info("Generating 'zero' indicator column")
        df['zero'] = (df['y'] == 0).astype(int)
    
    # Subsample if needed
    if subsample_size is not None and subsample_size < len(df):
        logger.info(f"Subsampling {subsample_size:,} rows from {len(df):,} total")
        df = df.sample(subsample_size, random_state=42)
    
    logger.debug(f"Data loaded: {len(df):,} rows, {len(df.columns)} columns")
    logger.debug(f"Zero-inflation rate: {df['zero'].mean():.2%}")
    
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets, stratified by zero/non-zero.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    test_size : float
        Fraction of data to use for testing (between 0 and 1)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and testing DataFrames
    """
    logger.info(f"Splitting data with test_size={test_size}")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['zero']
    )
    
    logger.debug(f"Train set: {len(train_df):,} rows")
    logger.debug(f"Test set: {len(test_df):,} rows")
    
    return train_df, test_df


def get_feature_target_arrays(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature and target arrays from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    feature_cols : List[str], optional
        List of column names to use as features.
        If None, uses ['X1', 'X2', 'X3'] by default.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        X (features), y_zero (binary target), y_count (count target)
    """
    if feature_cols is None:
        feature_cols = ['X1', 'X2', 'X3']
        
    logger.debug(f"Extracting features from columns: {feature_cols}")
    
    X = df[feature_cols].values.astype(np.float32)
    y_zero = df['zero'].values
    y_count = df['y'].values
    
    return X, y_zero, y_count


def load_and_split_data(config: Config) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Load and split data according to configuration.
    
    Parameters
    ----------
    config : Config
        Configuration object containing data settings
        
    Returns
    -------
    Dict[str, Union[np.ndarray, pd.DataFrame]]
        Dictionary containing:
        - train_df: Training DataFrame
        - test_df: Testing DataFrame
        - X_train: Training features
        - y_zero_train: Training zero indicator
        - y_count_train: Training count values
        - X_test: Test features
        - y_zero_test: Test zero indicator
        - y_count_test: Test count values
    """
    # Load data
    df = load_data(
        file_path=config.data.input_file,
        subsample_size=config.data.subsample_size
    )
    
    # Split data
    train_df, test_df = split_data(
        df=df,
        test_size=config.data.test_size,
        random_state=config.general.random_seed
    )
    
    # Get arrays
    X_train, y_zero_train, y_count_train = get_feature_target_arrays(train_df)
    X_test, y_zero_test, y_count_test = get_feature_target_arrays(test_df)
    
    # Return all components
    return {
        'train_df': train_df,
        'test_df': test_df,
        'X_train': X_train,
        'y_zero_train': y_zero_train,
        'y_count_train': y_count_train,
        'X_test': X_test,
        'y_zero_test': y_zero_test,
        'y_count_test': y_count_test,
    }


if __name__ == "__main__":
    # Set up logging for stand-alone execution
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    from config import load_config
    
    try:
        config = load_config("config.yaml")
        data_dict = load_and_split_data(config)
        
        print(f"Loaded and split data successfully.")
        print(f"Training data: {len(data_dict['train_df']):,} rows")
        print(f"Testing data: {len(data_dict['test_df']):,} rows")
        print(f"Feature matrix shape: {data_dict['X_train'].shape}")
        print(f"Zero rate in training: {data_dict['y_zero_train'].mean():.2%}")
    except Exception as e:
        print(f"Error in data module: {str(e)}")
