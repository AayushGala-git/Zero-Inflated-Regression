"""
Evaluation module for zero-inflated count models.

This module provides functions for evaluating model performance,
generating visualizations, and performing model diagnostics.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from scipy.stats import genpareto, lognorm
from statsmodels.graphics.gofplots import qqplot
import os
from config import Config
from typing import Dict, Optional, List, Any

# Set up module logger
logger = logging.getLogger(__name__)


def evaluate_zero_stage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate zero-stage (classification) model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1)
    y_prob : np.ndarray
        Predicted probabilities of the positive class (1)
    threshold : float
        Classification threshold for converting probabilities to binary predictions
        
    Returns
    -------
    Dict[str, float]
        Dictionary with evaluation metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'auc': roc_auc_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    metrics.update({
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
    })
    
    logger.info(f"Zero-stage evaluation: AUC={metrics['auc']:.4f}, Accuracy={metrics['accuracy']:.4f}")
    
    return metrics


def evaluate_count_stage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate count-stage (regression) model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True count values
    y_pred : np.ndarray
        Predicted count values
    zero_mask : np.ndarray, optional
        Mask for zero values to evaluate separately
        
    Returns
    -------
    Dict[str, float]
        Dictionary with evaluation metrics
    """
    # Overall metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }
    
    logger.info(f"Count-stage evaluation: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    # Metrics on positive counts only
    pos_mask = y_true > 0
    if pos_mask.sum() > 0:
        metrics.update({
            'rmse_pos': np.sqrt(mean_squared_error(y_true[pos_mask], y_pred[pos_mask])),
            'mae_pos': mean_absolute_error(y_true[pos_mask], y_pred[pos_mask]),
            'r2_pos': r2_score(y_true[pos_mask], y_pred[pos_mask]) if pos_mask.sum() > 1 else float('nan'),
        })
        logger.info(f"Positive counts: RMSE={metrics['rmse_pos']:.4f}, MAE={metrics['mae_pos']:.4f}")
    
    # If zero mask provided, evaluate on zeros and non-zeros separately
    if zero_mask is not None:
        # Zero counts
        if zero_mask.sum() > 0:
            metrics.update({
                'rmse_zeros': np.sqrt(mean_squared_error(y_true[zero_mask], y_pred[zero_mask])),
                'mae_zeros': mean_absolute_error(y_true[zero_mask], y_pred[zero_mask]),
            })
        
        # Non-zero counts
        non_zero_mask = ~zero_mask
        if non_zero_mask.sum() > 0:
            metrics.update({
                'rmse_nonzeros': np.sqrt(mean_squared_error(y_true[non_zero_mask], y_pred[non_zero_mask])),
                'mae_nonzeros': mean_absolute_error(y_true[non_zero_mask], y_pred[non_zero_mask]),
                'r2_nonzeros': r2_score(y_true[non_zero_mask], y_pred[non_zero_mask]) if non_zero_mask.sum() > 1 else float('nan'),
            })
    
    return metrics


def evaluate_combined_model(
    y_true: np.ndarray,
    y_zero_true: np.ndarray,
    y_pred: np.ndarray,
    y_zero_prob: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate the combined zero-inflated model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True count values
    y_zero_true : np.ndarray
        True binary zero indicator
    y_pred : np.ndarray
        Predicted count values
    y_zero_prob : np.ndarray
        Predicted zero probabilities
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with evaluation metrics for zero-stage, count-stage, and overall
    """
    # Evaluate zero-stage
    zero_metrics = evaluate_zero_stage(y_zero_true, y_zero_prob)
    
    # Evaluate count-stage
    count_metrics = evaluate_count_stage(y_true, y_pred, zero_mask=(y_zero_true==1))
    
    # Overall metrics
    overall_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }
    
    logger.info(f"Overall evaluation: RMSE={overall_metrics['rmse']:.4f}, MAE={overall_metrics['mae']:.4f}")
    
    return {
        'zero_stage': zero_metrics,
        'count_stage': count_metrics,
        'overall': overall_metrics
    }


def evaluate_single_model(model, X_test, y_test) -> Dict[str, float]:
    """
    Evaluates a single regression model.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
    }
    return metrics


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration curve for zero-stage model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0/1)
    y_prob : np.ndarray
        Predicted probabilities of the positive class (1)
    n_bins : int
        Number of bins for calibration curve
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with the calibration curve
    """
    logger.info(f"Generating calibration curve with {n_bins} bins")
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot calibration curve
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label="Model")
    
    # Plot ideal calibration (diagonal)
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
    
    # Add labels and legend
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_title("Calibration Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if output_path provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved calibration curve to {output_path}")
    
    return fig


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_mask: Optional[np.ndarray] = None,
    max_value: Optional[int] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot predicted vs actual counts.
    
    Parameters
    ----------
    y_true : np.ndarray
        True count values
    y_pred : np.ndarray
        Predicted count values
    zero_mask : np.ndarray, optional
        Mask for zero values to plot separately
    max_value : int, optional
        Maximum value for axes limits
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with the predicted vs actual plot
    """
    logger.info("Generating predicted vs actual plot")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # If zero_mask is provided, separate zeros and non-zeros
    if zero_mask is not None and zero_mask.sum() > 0:
        # Plot non-zeros
        non_zero_mask = ~zero_mask
        ax.scatter(y_true[non_zero_mask], y_pred[non_zero_mask], 
                   alpha=0.5, s=10, label="Non-zeros")
        
        # Plot zeros
        ax.scatter(y_true[zero_mask], y_pred[zero_mask], 
                   alpha=0.5, s=10, c='red', label="Zeros")
    else:
        # Plot all points
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # Add diagonal line
    if max_value is not None:
        ax.plot([0, max_value], [0, max_value], 'k--', alpha=0.7)
    else:
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.7)
    
    # Set axis limits if max_value provided
    if max_value is not None:
        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)
    
    # Add labels and legend
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual Counts")
    ax.grid(True, alpha=0.3)
    if zero_mask is not None:
        ax.legend()
    
    # Save if output_path provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved predicted vs actual plot to {output_path}")
    
    return fig


def plot_qq_tail(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_percentile: float = 95,
    output_path: Optional[str] = None,
    dist: str = 'lognorm'
) -> plt.Figure:
    """
    Plot Q-Q plot for the tail of the distribution.
    
    Parameters
    ----------
    y_true : np.ndarray
        True count values
    y_pred : np.ndarray
        Predicted count values
    threshold_percentile : float
        Percentile threshold for defining the tail
    output_path : str, optional
        Path to save the plot
    dist : str
        Distribution to use for QQ plot ('lognorm' or 'genpareto')
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with the Q-Q plot
    """
    logger.info(f"Generating Q-Q plot for tail (>{threshold_percentile}th percentile)")
    
    # Filter to positive counts
    pos_mask = y_true > 0
    y_true_pos = y_true[pos_mask]
    y_pred_pos = y_pred[pos_mask]
    
    # Define threshold
    threshold = np.percentile(y_true_pos, threshold_percentile)
    
    # Filter to tail values
    tail_mask = y_true_pos > threshold
    y_true_tail = y_true_pos[tail_mask]
    y_pred_tail = y_pred_pos[tail_mask]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # QQ plot for true tail values
    if dist == 'genpareto':
        # Fit GPD to data
        shape, loc, scale = genpareto.fit(y_true_tail)
        qqplot(y_true_tail, genpareto, distargs=(shape,), loc=loc, scale=scale, line='45', ax=ax1)
    else:
        # Default to lognormal but fit parameters first
        shape, loc, scale = lognorm.fit(y_true_tail)
        qqplot(y_true_tail, lognorm, distargs=(shape,), loc=loc, scale=scale, line='45', ax=ax1)
    
    ax1.set_title(f"Q-Q Plot: True Tail Values (>{threshold:.1f})")
    
    # QQ plot for predicted tail values
    if dist == 'genpareto':
        # Fit GPD to data
        shape, loc, scale = genpareto.fit(y_pred_tail)
        qqplot(y_pred_tail, genpareto, distargs=(shape,), loc=loc, scale=scale, line='45', ax=ax2)
    else:
        # Default to lognormal but fit parameters first
        shape, loc, scale = lognorm.fit(y_pred_tail)
        qqplot(y_pred_tail, lognorm, distargs=(shape,), loc=loc, scale=scale, line='45', ax=ax2)
    
    ax2.set_title(f"Q-Q Plot: Predicted Tail Values")
    
    plt.tight_layout()
    
    # Save if output_path provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved tail Q-Q plot to {output_path}")
    
    return fig


def plot_loss_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Parameters
    ----------
    train_losses : List[float]
        List of training losses
    val_losses : List[float], optional
        List of validation losses
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with the loss curves
    """
    logger.info("Generating loss curve plot")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training loss
    ax.plot(train_losses, label="Training loss")
    
    # Plot validation loss if provided
    if val_losses is not None:
        ax.plot(val_losses, label="Validation loss")
    
    # Add labels and legend
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if output_path provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved loss curve to {output_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots the ROC curve.
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {output_path}")
        
    return fig


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_zero_true: np.ndarray,
    y_zero_prob: np.ndarray,
    optimal_threshold: float,
    model_name: str,
    zero_model: Any,
    count_model: Any,
    feature_names: List[str]
) -> str:
    """
    Generate a detailed evaluation report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True count values
    y_pred : np.ndarray
        Final predicted count values
    y_zero_true : np.ndarray
        True binary zero indicator
    y_zero_prob : np.ndarray
        Predicted probabilities for the zero class
    optimal_threshold : float
        The threshold used to classify zero vs. non-zero
    model_name : str
        Name of the model being evaluated
    zero_model : Any
        The trained zero-stage model object
    count_model : Any
        The trained count-stage model object
    feature_names : List[str]
        List of feature names used in the models

    Returns
    -------
    str
        Report text
    """
    logger.info("Generating evaluation report")

    # --- Zero-Stage Evaluation ---
    zero_metrics = evaluate_zero_stage(y_zero_true, y_zero_prob, threshold=optimal_threshold)

    # --- Count-Stage Evaluation (on positive values) ---
    pos_mask_true = y_true > 0
    pos_mask_pred = y_pred > 0
    
    # Evaluate only where both true and predicted values are positive
    eval_mask = pos_mask_true & pos_mask_pred
    if np.sum(eval_mask) > 0:
        count_metrics = evaluate_count_stage(y_true[eval_mask], y_pred[eval_mask])
    else:
        count_metrics = {
            'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan'),
            'rmse_pos': float('nan'), 'mae_pos': float('nan'), 'r2_pos': float('nan')
        }

    # --- Overall Evaluation ---
    overall_metrics = evaluate_count_stage(y_true, y_pred)

    # Generate report text
    report = []
    report.append("=" * 80)
    report.append(f"EVALUATION REPORT: {model_name}")
    report.append("=" * 80)
    report.append(f"\n--- Zero-Stage (Classification) Metrics (Threshold: {optimal_threshold:.4f}) ---")
    report.append(f"  AUC:         {zero_metrics.get('auc', 'N/A'):.4f}")
    report.append(f"  Accuracy:    {zero_metrics.get('accuracy', 'N/A'):.4f}")
    report.append(f"  Precision:   {zero_metrics.get('precision', 'N/A'):.4f}")
    report.append(f"  Recall:      {zero_metrics.get('recall', 'N/A'):.4f}")
    report.append(f"  F1 Score:    {zero_metrics.get('f1', 'N/A'):.4f}")

    report.append(f"\n--- Count-Stage (Regression) Metrics (on y_true > 0 & y_pred > 0) ---")
    report.append(f"  R-squared:   {count_metrics.get('r2', 'N/A'):.4f}")
    report.append(f"  RMSE:        {count_metrics.get('rmse', 'N/A'):.4f}")
    report.append(f"  MAE:         {count_metrics.get('mae', 'N/A'):.4f}")

    report.append(f"\n--- Overall Combined Model Metrics ---")
    report.append(f"  R-squared:   {overall_metrics.get('r2', 'N/A'):.4f}")
    report.append(f"  RMSE:        {overall_metrics.get('rmse', 'N/A'):.4f}")
    report.append(f"  MAE:         {overall_metrics.get('mae', 'N/A'):.4f}")
    
    report.append("\n" + "="*80 + "\n")
    
    return "\n".join(report)


def plot_distribution_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 50,
    max_count: Optional[int] = None,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a histogram comparing the distribution of actual vs predicted counts.
    
    Parameters
    ----------
    y_true : np.ndarray
        True count values
    y_pred : np.ndarray
        Predicted count values
    bins : int
        Number of bins for the histogram
    max_count : int, optional
        Maximum count value to display (for better visualization of the main distribution)
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with the distribution comparison
    """
    logger.info("Generating distribution comparison plot")
    
    # Create a copy with limited range for better visualization if specified
    if max_count is not None:
        y_true_plot = y_true[y_true <= max_count]
        y_pred_plot = y_pred[y_true <= max_count]  # Use same filter for consistency
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histograms
    ax.hist(y_true_plot, bins=bins, alpha=0.5, label="True Counts", density=True)
    ax.hist(y_pred_plot, bins=bins, alpha=0.5, label="Predicted Counts", density=True)
    
    # Add density curves (smoothed)
    from scipy.stats import gaussian_kde
    try:
        # Only try to plot density if we have enough unique values
        if len(np.unique(y_true_plot)) > 5:
            kde_true = gaussian_kde(y_true_plot)
            kde_pred = gaussian_kde(y_pred_plot)
            
            x = np.linspace(0, max(np.max(y_true_plot), np.max(y_pred_plot)), 1000)
            ax.plot(x, kde_true(x), 'b-', lw=2, label='True Density')
            ax.plot(x, kde_pred(x), 'r-', lw=2, label='Predicted Density')
    except:
        logger.warning("Could not generate density curves for distribution comparison")
    
    # Add descriptive statistics as text
    stats_text = f"True counts: Mean={np.mean(y_true):.2f}, Median={np.median(y_true):.2f}, Std={np.std(y_true):.2f}\n"
    stats_text += f"Predicted counts: Mean={np.mean(y_pred):.2f}, Median={np.median(y_pred):.2f}, Std={np.std(y_pred):.2f}"
    
    # Add statistics as text box
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel("Count Value")
    ax.set_ylabel("Density")
    ax.set_title("Distribution Comparison: True vs Predicted Counts")
    ax.legend()
    
    plt.tight_layout()
    
    # Save if output_path provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved distribution comparison plot to {output_path}")
    
    return fig


def plot_residuals(
    residuals: np.ndarray, 
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots the distribution of residuals.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title('Distribution of Residuals')
    ax.set_xlabel('Residual (True - Predicted)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved residuals plot to {output_path}")

    return fig


def plot_residuals_vs_predicted(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots residuals vs. predicted values.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Residuals vs. Predicted Values')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals (True - Predicted)')
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved residuals vs. predicted plot to {output_path}")

    return fig


def plot_feature_importance(
    feature_importance_df: pd.DataFrame, 
    output_path: str
):
    """
    Plots the feature importance.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    ax.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved feature importance plot to {output_path}")


def generate_visualizations(
    y_true_count: np.ndarray,
    y_pred_count: np.ndarray,
    y_true_binary: np.ndarray,
    y_prob_binary: np.ndarray,
    output_dir: str,
    prefix: str = ""
):
    """
    Generates and saves all standard visualizations.
    """
    logger.info(f"Generating all visualizations with prefix: '{prefix}'")

    # ROC Curve for zero-stage
    plot_roc_curve(
        y_true_binary, 
        y_prob_binary, 
        output_path=os.path.join(output_dir, f'{prefix}roc_curve.png')
    )

    # Predicted vs Actual for count-stage
    plot_pred_vs_actual(
        y_true_count, 
        y_pred_count, 
        output_path=os.path.join(output_dir, f'{prefix}pred_actual.png')
    )

    # Residuals plot
    residuals = y_true_count - y_pred_count
    plot_residuals(
        residuals, 
        output_path=os.path.join(output_dir, f'{prefix}residuals.png')
    )

    # Residuals vs. Predicted plot
    plot_residuals_vs_predicted(
        y_pred_count,
        residuals,
        output_path=os.path.join(output_dir, f'{prefix}residuals_vs_predicted.png')
    )
    
    logger.info("All visualizations generated.")


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_zero_true: np.ndarray,
    y_zero_prob: np.ndarray,
    optimal_threshold: float,
    model_name: str,
    zero_model: Any,
    count_model: Any,
    feature_names: List[str]
) -> str:
    """
    Generate a detailed evaluation report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True count values
    y_pred : np.ndarray
        Final predicted count values
    y_zero_true : np.ndarray
        True binary zero indicator
    y_zero_prob : np.ndarray
        Predicted probabilities for the zero class
    optimal_threshold : float
        The threshold used to classify zero vs. non-zero
    model_name : str
        Name of the model being evaluated
    zero_model : Any
        The trained zero-stage model object
    count_model : Any
        The trained count-stage model object
    feature_names : List[str]
        List of feature names used in the models

    Returns
    -------
    str
        Report text
    """
    logger.info("Generating evaluation report")

    # --- Zero-Stage Evaluation ---
    zero_metrics = evaluate_zero_stage(y_zero_true, y_zero_prob, threshold=optimal_threshold)

    # --- Count-Stage Evaluation (on positive values) ---
    pos_mask_true = y_true > 0
    pos_mask_pred = y_pred > 0
    
    # Evaluate only where both true and predicted values are positive
    eval_mask = pos_mask_true & pos_mask_pred
    if np.sum(eval_mask) > 0:
        count_metrics = evaluate_count_stage(y_true[eval_mask], y_pred[eval_mask])
    else:
        count_metrics = {
            'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan'),
            'rmse_pos': float('nan'), 'mae_pos': float('nan'), 'r2_pos': float('nan')
        }

    # --- Overall Evaluation ---
    overall_metrics = evaluate_count_stage(y_true, y_pred)

    # Generate report text
    report = []
    report.append("=" * 80)
    report.append(f"EVALUATION REPORT: {model_name}")
    report.append("=" * 80)
    report.append(f"\n--- Zero-Stage (Classification) Metrics (Threshold: {optimal_threshold:.4f}) ---")
    report.append(f"  AUC:         {zero_metrics.get('auc', 'N/A'):.4f}")
    report.append(f"  Accuracy:    {zero_metrics.get('accuracy', 'N/A'):.4f}")
    report.append(f"  Precision:   {zero_metrics.get('precision', 'N/A'):.4f}")
    report.append(f"  Recall:      {zero_metrics.get('recall', 'N/A'):.4f}")
    report.append(f"  F1 Score:    {zero_metrics.get('f1', 'N/A'):.4f}")

    report.append(f"\n--- Count-Stage (Regression) Metrics (on y_true > 0 & y_pred > 0) ---")
    report.append(f"  R-squared:   {count_metrics.get('r2', 'N/A'):.4f}")
    report.append(f"  RMSE:        {count_metrics.get('rmse', 'N/A'):.4f}")
    report.append(f"  MAE:         {count_metrics.get('mae', 'N/A'):.4f}")

    report.append(f"\n--- Overall Combined Model Metrics ---")
    report.append(f"  R-squared:   {overall_metrics.get('r2', 'N/A'):.4f}")
    report.append(f"  RMSE:        {overall_metrics.get('rmse', 'N/A'):.4f}")
    report.append(f"  MAE:         {overall_metrics.get('mae', 'N/A'):.4f}")
    
    report.append("\n" + "="*80 + "\n")
    
    return "\n".join(report)