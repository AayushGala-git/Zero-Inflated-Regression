# Zero-Inflated Model Analysis Pipeline

This project implements a comprehensive pipeline to model zero-inflated count data. It systematically explores, evaluates, and compares a two-stage (Hurdle) model against a simpler, single-regressor benchmark. The final analysis concludes that the benchmark model is superior for this dataset.

## Key Findings & Conclusion

After extensive experimentation, including advanced feature engineering and rigorous hyperparameter tuning, the key finding is:

**The single LightGBM Regressor (Benchmark Model) significantly outperforms the more complex Two-Stage Hurdle Model.**

| Metric | Benchmark Model (LGBM) | Two-Stage Model |
| :--- | :--- | :--- |
| **RMSE** | **3.23** | 11.98 |
| **MAE** | **1.08** | 7.77 |
| **R²** | **0.095** | -11.43 |

The benchmark model is not only more accurate but also simpler, faster to train, and easier to maintain. The two-stage approach, while theoretically sound for zero-inflated data, failed to generalize effectively on this specific dataset, leading to poor overall performance.

## Classification Performance Test

To specifically evaluate the model's ability to distinguish between zero and non-zero outcomes, a separate classification test was conducted on a newly generated dataset (`classification_data.csv`).

**The model achieved an accuracy of 73.80% on this task.**

| Metric | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Zero** | 0.96 | 0.72 | 0.82 |
| **Non-Zero**| 0.35 | 0.83 | 0.49 |

This shows that while the model is effective at identifying the majority of non-zero cases (high recall), it does so at the cost of making many false positive predictions (low precision).

## Final Project Structure

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

- Python 3.8+
- See `requirements.txt` for the full list of packages.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### Running the Pipeline

1.  **(Optional) Generate New Data:**
    If you need to generate a new synthetic dataset, run:
    ```bash
    python data_generation.py
    ```
    This will create `zero_inflated_data.csv` based on the parameters in `config.yaml`.

2.  **(Optional) Tune Hyperparameters:**
    To find the optimal hyperparameters for either the benchmark or the two-stage model components, use the `tune_models.py` script.

    *   **Tune the benchmark regressor:**
        ```bash
        python tune_models.py --model count
        ```
        This will find the best parameters for the single regressor and save them to `outputs/best_count_hyperparameters.json`.

    *   **Tune the zero-stage classifier:**
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
