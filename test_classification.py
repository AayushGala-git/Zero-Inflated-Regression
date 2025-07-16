"""
This script is dedicated to evaluating the classification performance of the model.
It uses a newly generated dataset to test how well the model can distinguish
between zero and non-zero outcomes.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
import json
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data import load_data
from advanced_feature_engineering import AdvancedFeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_classification_test(config_path):
    """
    Runs the full classification test pipeline.
    """
    logging.info("========================================================")
    logging.info("      STARTING CLASSIFICATION PERFORMANCE TEST")
    logging.info("========================================================")

    # 1. Load Configuration
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load Data
    data_path = config['data']['path']
    logging.info(f"Loading new dataset from {data_path}")
    df = load_data(data_path)

    # 3. Prepare Data for Classification
    logging.info("Preparing data for binary classification (zero vs. non-zero)")
    X = df[config['data']['features']]
    y = (df[config['data']['target']] > 0).astype(int) # Target is 1 if count > 0, else 0

    # 4. Split Data
    train_config = config['training']
    logging.info(f"Splitting data with test_size={train_config['test_size']} and random_state={train_config['random_state']}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=train_config['test_size'], 
        random_state=train_config['random_state'],
        stratify=y
    )

    # 5. Apply Feature Engineering
    logging.info("Applying advanced feature engineering...")
    fe_config = config['feature_engineering']
    feature_engineer = AdvancedFeatureEngineer(**fe_config)
    
    X_train_eng = feature_engineer.fit_transform(X_train)
    X_test_eng = feature_engineer.transform(X_test)
    logging.info(f"Data transformed. Number of features: {X_train_eng.shape[1]}")

    # 6. Load Hyperparameters
    hyperparams_path = train_config['hyperparameters_path']
    logging.info(f"Loading hyperparameters from {hyperparams_path}")
    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)

    # Extract the correct set of parameters for the zero-stage model
    # This handles both nested and flat hyperparameter files
    zero_stage_hyperparams = hyperparams.get('zero_stage', hyperparams)

    # 7. Train LGBM Classifier
    logging.info("Training LightGBM Classifier...")
    # Ensure scale_pos_weight is calculated correctly for the training data
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1
    zero_stage_hyperparams['scale_pos_weight'] = scale_pos_weight

    model = lgb.LGBMClassifier(**zero_stage_hyperparams)
    model.fit(X_train_eng, y_train)
    logging.info("Model training complete.")

    # 8. Evaluate Model
    logging.info("Evaluating model on the test set...")
    y_pred = model.predict(X_test_eng)

    # --- METRICS ---
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Zero', 'Non-Zero'])
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info("\n\n--- CLASSIFICATION TEST RESULTS ---")
    logging.info(f"\nAccuracy Score: {accuracy:.4f}\n")
    logging.info("Classification Report:")
    print(report)
    logging.info("Confusion Matrix:")
    print(conf_matrix)
    logging.info("\n--- TEST COMPLETE ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a classification test on a dedicated dataset.")
    parser.add_argument('--config', type=str, default='config_classification.yaml',
                        help='Path to the configuration file for the classification test.')
    args = parser.parse_args()
    run_classification_test(config_path=args.config)
