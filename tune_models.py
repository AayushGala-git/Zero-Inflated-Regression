"""
This script handles hyperparameter tuning for both the zero-inflation (classifier)
and the count (regressor) models using Optuna.

Which model to tune is specified via command-line arguments.
"""
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
import logging
import yaml
import json
import argparse
from data import load_data
from advanced_feature_engineering import AdvancedFeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def objective_zero(trial, X_train, y_train, X_val, y_val, early_stopping_rounds):
    """Optuna objective function for the zero-inflation (binary classification) model."""
    # Calculate scale_pos_weight for handling class imbalance
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1

    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'scale_pos_weight': scale_pos_weight,
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = lgb.LGBMClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc

def objective_count(trial, X_train, y_train, X_val, y_val, early_stopping_rounds):
    """Optuna objective function for the count (regression) model using Quantile Regression."""
    param = {
        'objective': 'quantile',  # Use Quantile Regression for robustness to outliers
        'alpha': 0.5,             # Target the median (50th percentile)
        'metric': 'quantile',     # Optimize for the quantile loss
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = lgb.LGBMRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

def run_tuning(config_path, model_type):
    """Main function to run hyperparameter tuning for the specified model type."""
    logging.info(f"Starting hyperparameter tuning for the '{model_type}' model.")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load and preprocess data
    logging.info("Loading and engineering features...")
    df = load_data(config['data']['path'])
    
    fe_config = config.get('feature_engineering', {})
    feature_engineer = AdvancedFeatureEngineer(**fe_config)
    
    X_original = df[config['data']['features']]
    processed_df_values = feature_engineer.fit_transform(X_original)
    
    processed_df = pd.DataFrame(processed_df_values, columns=feature_engineer.get_feature_names(), index=df.index)
    processed_df['y'] = df['y']

    # Prepare data based on model type
    if model_type == 'zero':
        logging.info("Preparing data for zero-inflation model.")
        X = processed_df[feature_engineer.get_feature_names()]
        y = (processed_df['y'] == 0).astype(int)
        objective_fn = objective_zero
        direction = 'maximize'
        output_key = 'zero_hyperparameter_path'
    elif model_type == 'count':
        logging.info("Filtering for non-zero count data.")
        count_df = processed_df[processed_df['y'] > 0].copy()
        logging.info(f"Found {len(count_df)} non-zero samples for tuning.")
        X = count_df[feature_engineer.get_feature_names()]
        y = count_df['y']
        objective_fn = objective_count
        direction = 'minimize'
        output_key = 'count_hyperparameter_path'
    else:
        raise ValueError("model_type must be 'zero' or 'count'")

    # Split data
    train_config = config.get('training', {})
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=train_config.get('random_state', 42), stratify=(y if model_type == 'zero' else None)
    )

    # Run Optuna Study
    logging.info(f"Starting Optuna study (direction: {direction})...")
    study = optuna.create_study(direction=direction)
    
    tuning_config = config.get('tuning', {})
    early_stopping_rounds = tuning_config.get('early_stopping_rounds', 50)
    n_trials = tuning_config.get('n_trials', 50)
    
    study.optimize(lambda trial: objective_fn(trial, X_train, y_train, X_val, y_val, early_stopping_rounds), 
                   n_trials=n_trials)

    logging.info(f"Best trial for {model_type} model: {study.best_trial.value}")
    logging.info(f"Best params: {study.best_params}")

    # Save the best hyperparameters
    output_path = tuning_config.get(output_key)
    with open(output_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logging.info(f"Best hyperparameters for {model_type} model saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tune model hyperparameters.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--model', type=str, required=True, choices=['zero', 'count'], help="Which model to tune: 'zero' or 'count'.")
    args = parser.parse_args()
    
    run_tuning(config_path=args.config, model_type=args.model)
