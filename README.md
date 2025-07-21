# Zero-Inflated Regression: A Comparative Study

## Project Goal

This project investigates the most effective modeling strategy for zero-inflated count data, a common challenge in fields like health sciences and insurance. The primary goal was to test the hypothesis that a complex, two-stage "Hurdle" model, which explicitly models the zero-generating process, would outperform a simpler, single-stage regression model.

## Final Conclusion: Simpler is Better

The core finding of this project is that **the hypothesis was disproven**. For this dataset, a single, robust Quantile Regression model significantly outperformed the more complex two-stage model.

The reason for this outcome is **error propagation**. While the first stage of the Hurdle model (the classifier) was reasonably good at identifying zeros, its inevitable mistakes were magnified in the second stage, leading to large prediction errors. The simpler benchmark model, by handling all data at once, proved to be more robust and accurate.

### Final Performance Metrics

| Metric                | Two-Stage Hurdle Model | **Benchmark (Single Regressor)** |
| --------------------- | ---------------------- | -------------------------------- |
| **Overall MAE**       | 5.73                   | **0.86 (Winner)**                |
| **Overall RMSE**      | 8.79                   | **3.40 (Winner)**                |
| **Overall R-squared** | -5.69                  | -0.002                           |

This result is the central conclusion for a dissertation: while theoretically appealing, a two-stage model is not always the best practical solution, and its complexity can be a critical point of failure.

## Repository Structure

The project is organized into a modular, configuration-driven pipeline. All generated artifacts are saved in the `outputs/` directory.

```text
.
├── config.yaml                     # Central configuration for the pipeline
├── train.py                        # Main script to train and evaluate models
├── evaluate.py                     # Evaluation logic and plotting functions
├── tune_models.py                  # Hyperparameter tuning script
├── requirements.txt                # Project dependencies
│
└── outputs/                        # All generated files are stored here
    ├── reports/
    │   └── evaluation_report.txt   # Detailed performance metrics for both models
    ├── plots/
    │   ├── benchmark_*.png         # All plots for the benchmark model
    │   └── two_stage_*.png         # All plots for the two-stage model
    ├── models/
    │   ├── benchmark_model.pkl
    │   └── two_stage_*.pkl
    └── hyperparams/
        ├── best_count_hyperparameters.json
        └── best_zero_hyperparameters.json
```

## How to Run the Pipeline

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/AayushGala-git/Zero-Inflated-Regression.git
cd Zero-Inflated-Regression

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. (Optional) Tune Hyperparameters

Run the tuning script to find the best parameters for each model component.

```bash
# Tune the classifier (for the two-stage model)
python tune_models.py --model zero

# Tune the regressor (for the benchmark model and second stage)
python tune_models.py --model count
```

This will save `best_zero_hyperparameters.json` and `best_count_hyperparameters.json` in `outputs/hyperparams/`.

### 3. Run the Main Training Pipeline

This is the primary script that trains both models, evaluates them, and generates all reports, plots, and model files.

```bash
python train.py --config config.yaml
```

All outputs will be saved to the `outputs/` directory, organized by type.
