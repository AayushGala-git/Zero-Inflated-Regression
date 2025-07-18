# Project Title: Zero-Inflated Count Data Modeling

## Overview

This project explores various modeling techniques for zero-inflated count data. The primary goal is to build a predictive model that can accurately handle a large number of zero values while also correctly predicting the magnitude of non-zero counts.

The core of the project involves comparing a two-stage (Hurdle) model against a single robust regression model (benchmark).

## Final Results & Conclusion

After extensive experimentation, including feature engineering, hyperparameter tuning, and model architecture changes, we have arrived at a clear conclusion.

### Model Architectures Tested

1. **Two-Stage Hurdle Model:**
    * **Stage 1 (Classification):** A `lightgbm.LGBMClassifier` predicts the probability of a transaction being non-zero (`P(y > 0)`).
    * **Stage 2 (Regression):** A `lightgbm.LGBMRegressor` with a **quantile regression objective** predicts the count value for transactions identified as non-zero (`E[y | y > 0]`).

2. **Benchmark Model:**
    * A single `lightgbm.LGBMRegressor`, also using a **quantile regression objective**, trained on the entire dataset.

### Key Findings

| Metric                      | Two-Stage Hurdle Model | Benchmark (Single Regressor) |
| --------------------------- | ---------------------- | ---------------------------- |
| **Overall R-squared**       | -5.69                  | -0.002                       |
| **Overall MAE**             | 5.73                   | **0.86**                     |
| **Classifier AUC (Stage 1)**| 0.85                   | N/A                          |
| **Regressor R² (Stage 2)**  | 0.27                   | N/A                          |

1. **Quantile Regression Was Effective:** The decision to switch the regression component to use a quantile objective was successful. The regressor in the two-stage model achieved a respectable R-squared of **0.27** on the subset of data it was trained on.

2. **Hurdle Model Fails Due to Error Propagation:** Despite the success of the regression stage, the overall two-stage model performed very poorly (Overall R² of -5.69). The failure lies in the classification stage. While it had high precision (0.93), its recall was only 0.68, meaning it incorrectly classified 32% of non-zero transactions as zero. These classification errors introduced massive prediction errors that the regression stage could not overcome.

3. **Simpler is Better:** The benchmark model, a single robust regressor, proved to be the superior approach. Although its R-squared was near zero, its Mean Absolute Error (MAE) of **0.86** was nearly **7 times lower** than the hurdle model's MAE of 5.73. This indicates that its predictions are significantly more accurate and reliable in a real-world scenario.

### Final Conclusion

For this dataset, the complexity of a two-stage hurdle model is detrimental. A single, robust regression model (like LightGBM with a quantile or Tweedie objective) that can inherently handle zero values is the more effective and reliable strategy. The attempt to perfectly separate zeros from non-zeros creates more error than it solves.

## Project Structure

The repository is organized into a modular and scalable pipeline:

```text
.
├── config.yaml                        # Central configuration for the main pipeline
├── config_classification.yaml         # Configuration for the classification test
├── data_generation.py                 # Script to create synthetic data
├── train.py                           # Main script for the regression task
├── test_classification.py             # Script to run the classification test
├── tune_models.py                     # Script for hyperparameter tuning
├── evaluate.py                        # Contains all evaluation and visualization logic
├── data.py                            # Data loading utilities
├── advanced_feature_engineering.py    # Advanced feature creation logic
├── requirements.txt                   # Project dependencies
├── outputs/                           # Directory for all generated reports, plots, and model files
│   ├── evaluation_report.txt          # Detailed performance metrics
│   └── ... (other plots)
└── tests/                             # Unit tests for key components
```

## How to Run the Pipeline

### Requirements

* Python 3.8+
* See `requirements.txt` for the full list of packages.

### Installation

1. **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <repository-name>
    ```

2. **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### Running the Pipeline

1. **(Optional) Generate New Data:**
    If you need to generate a new synthetic dataset, run:

    ```bash
    python data_generation.py
    ```

    This will create `zero_inflated_data.csv` based on the parameters in `config.yaml`.

2. **(Optional) Tune Hyperparameters:**
    To find the optimal hyperparameters for either the benchmark or the two-stage model components, use the `tune_models.py` script.

    * **Tune the benchmark regressor:**

        ```bash
        python tune_models.py --model count
        ```

        This will find the best parameters for the single regressor and save them to `outputs/best_count_hyperparameters.json`.

    * **Tune the zero-stage classifier:**

        ```bash
        python tune_models.py --model zero
        ```

        This saves results to `outputs/best_zero_hyperparameters.json`.

3.  **Run the Main Training and Evaluation Pipeline:**
    This is the primary script that trains the models, evaluates them, and generates all the analysis plots and reports.

    ```bash
    python train.py --config config.yaml
    ```

    The script will use the hyperparameters defined in `config.yaml`. For best results, update the config with the parameters found during the tuning step.

4.  **Run the Classification Test:**
    To evaluate the model's ability to classify zero vs. non-zero outcomes on a separate dataset, run:

    ```bash
    python test_classification.py --config config_classification.yaml
    ```

    This script uses its own configuration and dataset to provide an independent accuracy score.

## What We Improved

This project evolved from a basic script into a robust pipeline:

1.  **Advanced Feature Engineering**: Automated the creation of polynomial, interaction, and statistical features, moving beyond the initial basic set.
2.  **Systematic Tuning**: Replaced manual tuning with an `Optuna`-based approach, allowing for efficient and reproducible hyperparameter searches.
3.  **Comparative Analysis**: Implemented a benchmark model to provide a crucial baseline, which ultimately proved to be the better solution.
4.  **Comprehensive Evaluation**: Generated a suite of visualizations for in-depth error analysis, including feature importance, residual plots, and predicted vs. actual plots.
5.  **Modular & Configurable Code**: Refactored the entire codebase into a clean, modular structure managed by a central `config.yaml` file, making it easy to modify and run.
