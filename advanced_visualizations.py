"""
Advanced visualization module for zero-inflated count models.

This module provides sophisticated plotting functions inspired by academic research
on zero-inflated models, including diagnostic plots, residual analysis, 
distribution comparisons, and model interpretation visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, poisson, nbinom
from sklearn.metrics import r2_score
import os
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_comprehensive_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_prob: np.ndarray,
    y_zero_true: np.ndarray,
    output_dir: str,
    dpi: int = 300
):
    """
    Create comprehensive diagnostic plots for zero-inflated models.
    
    Includes:
    1. Zero-inflation diagnostic plot
    2. Count distribution comparison (observed vs predicted)
    3. Pearson residuals analysis
    4. Deviance residuals plot
    5. Rootogram (hanging histogram)
    6. Probability integral transform (PIT) histogram
    """
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Comprehensive Zero-Inflated Model Diagnostics', fontsize=16, fontweight='bold')
    
    # 1. Zero-inflation diagnostic
    ax = axes[0, 0]
    zero_pred = (zero_prob > 0.5).astype(int)
    observed_zeros = np.sum(y_zero_true == 1)
    predicted_zeros = np.sum(zero_pred == 1)
    
    categories = ['Observed', 'Predicted']
    zero_counts = [observed_zeros, predicted_zeros]
    nonzero_counts = [len(y_true) - observed_zeros, len(y_true) - predicted_zeros]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, zero_counts, width, label='Zeros', alpha=0.8, color='lightcoral')
    ax.bar(x + width/2, nonzero_counts, width, label='Non-zeros', alpha=0.8, color='skyblue')
    
    ax.set_xlabel('Data Type')
    ax.set_ylabel('Count')
    ax.set_title('Zero vs Non-Zero Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add percentage annotations
    total = len(y_true)
    for i, (zeros, nonzeros) in enumerate(zip(zero_counts, nonzero_counts)):
        ax.text(i - width/2, zeros + total*0.01, f'{zeros/total:.1%}', 
                ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, nonzeros + total*0.01, f'{nonzeros/total:.1%}', 
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Count distribution comparison (for non-zero values)
    ax = axes[0, 1]
    nonzero_mask = (y_true > 0)
    y_true_nonzero = y_true[nonzero_mask]
    y_pred_nonzero = y_pred[nonzero_mask]
    
    # Create binned comparison
    max_count = min(max(y_true_nonzero.max(), y_pred_nonzero.max()), 20)  # Cap at 20 for readability
    bins = np.arange(1, max_count + 2) - 0.5
    
    ax.hist(y_true_nonzero, bins=bins, alpha=0.7, label='Observed', density=True, color='lightcoral')
    ax.hist(y_pred_nonzero, bins=bins, alpha=0.7, label='Predicted', density=True, color='skyblue')
    
    ax.set_xlabel('Count Value')
    ax.set_ylabel('Density')
    ax.set_title('Non-Zero Count Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Pearson residuals
    ax = axes[1, 0]
    # Calculate Pearson residuals for Poisson
    pearson_residuals = (y_true - y_pred) / np.sqrt(np.maximum(y_pred, 1e-6))
    
    ax.scatter(y_pred, pearson_residuals, alpha=0.6, s=20)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax.axhline(y=2, color='orange', linestyle='--', alpha=0.6)
    ax.axhline(y=-2, color='orange', linestyle='--', alpha=0.6)
    
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Pearson Residuals')
    ax.set_title('Pearson Residuals vs Predicted Values')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(y_pred, pearson_residuals, 1)
    p = np.poly1d(z)
    ax.plot(sorted(y_pred), p(sorted(y_pred)), "r-", alpha=0.8, linewidth=2)
    
    # 4. Deviance residuals
    ax = axes[1, 1]
    # Calculate deviance residuals
    deviance_residuals = np.sign(y_true - y_pred) * np.sqrt(
        2 * (y_true * np.log(np.maximum(y_true/np.maximum(y_pred, 1e-6), 1e-6)) - (y_true - y_pred))
    )
    deviance_residuals = np.nan_to_num(deviance_residuals)
    
    ax.scatter(y_pred, deviance_residuals, alpha=0.6, s=20)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax.axhline(y=2, color='orange', linestyle='--', alpha=0.6)
    ax.axhline(y=-2, color='orange', linestyle='--', alpha=0.6)
    
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Deviance Residuals')
    ax.set_title('Deviance Residuals vs Predicted Values')
    ax.grid(True, alpha=0.3)
    
    # 5. Rootogram (Hanging Histogram)
    ax = axes[2, 0]
    max_count_root = min(max(y_true.max(), y_pred.max()), 15)
    counts_obs = np.bincount(y_true.astype(int), minlength=max_count_root+1)[:max_count_root+1]
    counts_pred = np.bincount(np.round(y_pred).astype(int), minlength=max_count_root+1)[:max_count_root+1]
    
    x_vals = np.arange(len(counts_obs))
    sqrt_obs = np.sqrt(counts_obs)
    sqrt_pred = np.sqrt(counts_pred)
    
    # Hanging rootogram
    ax.bar(x_vals, sqrt_pred, alpha=0.7, color='skyblue', label='Predicted (sqrt scale)')
    ax.bar(x_vals, sqrt_obs - sqrt_pred, bottom=sqrt_pred, alpha=0.7, 
           color='lightcoral', label='Observed - Predicted')
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax.set_xlabel('Count Value')
    ax.set_ylabel('Square Root of Frequency')
    ax.set_title('Rootogram (Hanging Histogram)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Q-Q plot for standardized residuals
    ax = axes[2, 1]
    standardized_residuals = (y_true - y_pred) / np.sqrt(np.maximum(y_pred, 1))
    stats.probplot(standardized_residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Standardized Residuals vs Normal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_diagnostics.png'), 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def create_zero_inflation_analysis_plots(
    y_true: np.ndarray,
    zero_prob: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    dpi: int = 300
):
    """
    Create detailed zero-inflation analysis plots.
    
    Includes:
    1. Zero probability distribution
    2. Feature relationship with zero-inflation
    3. Zero-inflation probability calibration
    4. ROC curve for zero prediction
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Zero-Inflation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Zero probability distribution
    ax = axes[0, 0]
    zero_true_binary = (y_true == 0).astype(int)
    
    # Separate probabilities for true zeros and non-zeros
    prob_zeros = zero_prob[zero_true_binary == 1]
    prob_nonzeros = zero_prob[zero_true_binary == 0]
    
    ax.hist(prob_zeros, bins=50, alpha=0.7, label=f'True Zeros (n={len(prob_zeros)})', 
            density=True, color='lightcoral')
    ax.hist(prob_nonzeros, bins=50, alpha=0.7, label=f'True Non-zeros (n={len(prob_nonzeros)})', 
            density=True, color='skyblue')
    
    ax.set_xlabel('Predicted Zero Probability')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Zero Probabilities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Feature relationships with zero-inflation
    ax = axes[0, 1]
    if len(feature_names) >= 3:
        # Use first 3 features for visualization
        for i in range(min(3, len(feature_names))):
            feature_vals = features[:, i]
            # Bin the feature and calculate mean zero probability
            bins = pd.qcut(feature_vals, q=10, duplicates='drop')
            binned_data = pd.DataFrame({'feature': feature_vals, 'zero_prob': zero_prob, 'bins': bins})
            grouped = binned_data.groupby('bins')['zero_prob'].mean()
            
            bin_centers = [interval.mid for interval in grouped.index]
            ax.plot(bin_centers, grouped.values, marker='o', label=feature_names[i], linewidth=2)
    
    ax.set_xlabel('Feature Value (binned)')
    ax.set_ylabel('Mean Zero Probability')
    ax.set_title('Feature Relationships with Zero-Inflation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Calibration plot for zero prediction
    ax = axes[1, 0]
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        zero_true_binary, zero_prob, n_bins=10
    )
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(mean_predicted_value, fraction_of_positives, marker='o', 
            linewidth=2, label='Model Calibration')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Zero-Inflation Probability Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. ROC curve for zero prediction
    ax = axes[1, 1]
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(zero_true_binary, zero_prob)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for Zero Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zero_inflation_analysis.png'), 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def create_count_model_analysis_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    dpi: int = 300
):
    """
    Create count model specific analysis plots.
    
    Includes:
    1. Count vs predicted scatter with density
    2. Residuals by count value
    3. Over/under-dispersion analysis
    4. Count distribution comparison with statistical tests
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Count Model Analysis', fontsize=16, fontweight='bold')
    
    # Filter to non-zero values for count analysis
    nonzero_mask = (y_true > 0)
    y_true_nz = y_true[nonzero_mask]
    y_pred_nz = y_pred[nonzero_mask]
    
    # 1. Scatter plot with marginal distributions
    ax = axes[0, 0]
    
    # Create scatter plot with density coloring
    h = ax.hist2d(y_pred_nz, y_true_nz, bins=30, cmap='Blues', alpha=0.7)
    plt.colorbar(h[3], ax=ax, label='Frequency')
    
    # Add perfect prediction line
    max_val = max(y_true_nz.max(), y_pred_nz.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    
    ax.set_xlabel('Predicted Count')
    ax.set_ylabel('Observed Count')
    ax.set_title('Observed vs Predicted Counts (Non-zero)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Residuals by count value
    ax = axes[0, 1]
    residuals = y_true_nz - y_pred_nz
    
    # Bin by count value and show box plots
    count_bins = pd.qcut(y_true_nz, q=8, duplicates='drop')
    df_residuals = pd.DataFrame({'residuals': residuals, 'count_bins': count_bins})
    
    box_data = [group['residuals'].values for name, group in df_residuals.groupby('count_bins')]
    box_labels = [f'{interval.left:.1f}-{interval.right:.1f}' for interval in df_residuals['count_bins'].cat.categories]
    
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax.set_xlabel('Count Value Bins')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals by Count Value')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 3. Variance vs Mean plot (dispersion analysis)
    ax = axes[1, 0]
    
    # Bin predictions and calculate variance and mean for each bin
    pred_bins = pd.qcut(y_pred_nz, q=15, duplicates='drop')
    df_disp = pd.DataFrame({'observed': y_true_nz, 'predicted': y_pred_nz, 'bins': pred_bins})
    
    dispersion_stats = df_disp.groupby('bins').agg({
        'observed': ['mean', 'var'],
        'predicted': 'mean'
    }).round(3)
    
    obs_means = dispersion_stats[('observed', 'mean')].values
    obs_vars = dispersion_stats[('observed', 'var')].values
    
    ax.scatter(obs_means, obs_vars, alpha=0.7, s=50, color='blue', label='Observed')
    ax.plot([0, obs_means.max()], [0, obs_means.max()], 'r--', 
            label='Poisson (Var = Mean)', alpha=0.8)
    
    ax.set_xlabel('Mean Count')
    ax.set_ylabel('Variance')
    ax.set_title('Variance vs Mean (Dispersion Analysis)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Distribution comparison with fit statistics
    ax = axes[1, 1]
    
    # Create histogram comparison
    max_count = min(int(y_true_nz.max()), 20)
    bins = np.arange(1, max_count + 2) - 0.5
    
    hist_obs, _ = np.histogram(y_true_nz, bins=bins, density=True)
    hist_pred, _ = np.histogram(y_pred_nz, bins=bins, density=True)
    
    x_vals = np.arange(1, len(hist_obs) + 1)
    
    ax.bar(x_vals - 0.2, hist_obs, width=0.4, alpha=0.7, label='Observed', color='lightcoral')
    ax.bar(x_vals + 0.2, hist_pred, width=0.4, alpha=0.7, label='Predicted', color='skyblue')
    
    ax.set_xlabel('Count Value')
    ax.set_ylabel('Density')
    ax.set_title('Count Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add goodness-of-fit statistics as text
    from scipy.stats import chisquare, ks_2samp
    chi2_stat, chi2_p = chisquare(hist_obs, hist_pred)
    ks_stat, ks_p = ks_2samp(y_true_nz, y_pred_nz)
    
    textstr = f'χ² test: p={chi2_p:.3f}\\nKS test: p={ks_p:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'count_model_analysis.png'), 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def create_feature_analysis_plots(
    features: np.ndarray,
    feature_names: List[str],
    y_true: np.ndarray,
    zero_prob: np.ndarray,
    output_dir: str,
    dpi: int = 300
):
    """
    Create feature analysis and interpretation plots.
    
    Includes:
    1. Feature correlation heatmap
    2. Feature distributions by zero/non-zero
    3. Partial dependence plots
    4. Feature interaction effects
    """
    
    if len(feature_names) < 3:
        return  # Need at least 3 features for meaningful analysis
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Feature Analysis and Interpretation', fontsize=16, fontweight='bold')
    
    # Prepare data
    is_zero = (y_true == 0)
    feature_df = pd.DataFrame(features[:, :min(len(feature_names), 10)], 
                             columns=feature_names[:min(len(feature_names), 10)])
    
    # 1. Feature correlation heatmap
    ax = axes[0, 0]
    corr_matrix = feature_df.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title('Feature Correlation Matrix')
    
    # 2. Feature distributions by zero/non-zero
    ax = axes[0, 1]
    
    # Select top 3 most important features (by correlation with target)
    correlations = [np.corrcoef(features[:, i], is_zero.astype(int))[0, 1] 
                   for i in range(min(len(feature_names), features.shape[1]))]
    top_features_idx = np.argsort(np.abs(correlations))[-3:]
    
    for i, feat_idx in enumerate(top_features_idx):
        feature_vals = features[:, feat_idx]
        
        # Create density plots
        feat_zeros = feature_vals[is_zero]
        feat_nonzeros = feature_vals[~is_zero]
        
        ax.hist(feat_zeros, bins=30, alpha=0.6, density=True, 
                label=f'{feature_names[feat_idx]} (Zeros)', color=f'C{i}')
        ax.hist(feat_nonzeros, bins=30, alpha=0.6, density=True, 
                label=f'{feature_names[feat_idx]} (Non-zeros)', color=f'C{i}', 
                linestyle='--', histtype='step', linewidth=2)
    
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Density')
    ax.set_title('Feature Distributions by Zero/Non-zero')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Partial dependence plot (simplified)
    ax = axes[1, 0]
    
    # Use the most correlated feature with zero probability
    best_feat_idx = np.argmax(np.abs(correlations))
    feature_vals = features[:, best_feat_idx]
    
    # Bin the feature and calculate mean zero probability
    bins = pd.qcut(feature_vals, q=20, duplicates='drop')
    df_partial = pd.DataFrame({'feature': feature_vals, 'zero_prob': zero_prob, 'bins': bins})
    partial_effect = df_partial.groupby('bins')['zero_prob'].agg(['mean', 'std']).reset_index()
    
    bin_centers = [interval.mid for interval in partial_effect['bins']]
    
    ax.plot(bin_centers, partial_effect['mean'], marker='o', linewidth=2, 
            label=f'Mean Zero Probability', color='blue')
    ax.fill_between(bin_centers, 
                    partial_effect['mean'] - partial_effect['std'],
                    partial_effect['mean'] + partial_effect['std'],
                    alpha=0.3, color='blue')
    
    ax.set_xlabel(f'{feature_names[best_feat_idx]} Value')
    ax.set_ylabel('Zero Probability')
    ax.set_title(f'Partial Dependence: {feature_names[best_feat_idx]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Feature interaction effect (if we have at least 2 features)
    ax = axes[1, 1]
    
    if len(top_features_idx) >= 2:
        feat1_idx, feat2_idx = top_features_idx[-2:]
        feat1_vals = features[:, feat1_idx]
        feat2_vals = features[:, feat2_idx]
        
        # Create 2D binning
        feat1_bins = pd.qcut(feat1_vals, q=5, duplicates='drop')
        feat2_bins = pd.qcut(feat2_vals, q=5, duplicates='drop')
        
        interaction_df = pd.DataFrame({
            'feat1': feat1_vals, 'feat2': feat2_vals, 'zero_prob': zero_prob,
            'feat1_bins': feat1_bins, 'feat2_bins': feat2_bins
        })
        
        interaction_effect = interaction_df.groupby(['feat1_bins', 'feat2_bins'])['zero_prob'].mean().unstack()
        
        sns.heatmap(interaction_effect, annot=True, cmap='viridis', 
                   ax=ax, cbar_kws={"shrink": .8})
        ax.set_title(f'Interaction Effect: {feature_names[feat1_idx]} vs {feature_names[feat2_idx]}')
        ax.set_xlabel(feature_names[feat2_idx])
        ax.set_ylabel(feature_names[feat1_idx])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_analysis.png'), 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def create_model_comparison_summary(
    results_dict: Dict[str, Any],
    output_dir: str,
    dpi: int = 300
):
    """
    Create a comprehensive model comparison summary plot.
    
    Includes:
    1. Performance metrics comparison
    2. Prediction accuracy across count values
    3. Computational efficiency comparison
    4. Model complexity analysis
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
    
    # Extract metrics (this would need to be adapted based on your results structure)
    metrics = ['AUC', 'Accuracy', 'RMSE', 'MAE']
    
    # 1. Performance metrics radar chart
    ax = axes[0, 0]
    
    # This is a placeholder - would need actual model comparison data
    # For now, show the current model performance
    if 'metrics' in results_dict:
        metric_values = []
        for metric in metrics:
            if metric.lower() in results_dict['metrics']:
                metric_values.append(results_dict['metrics'][metric.lower()])
            else:
                metric_values.append(0.5)  # Default value
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        metric_values += metric_values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles, metric_values, 'o-', linewidth=2, label='Current Model')
        ax.fill(angles, metric_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics')
        ax.legend()
    
    # 2. Accuracy by count value
    ax = axes[0, 1]
    
    if 'predictions' in results_dict:
        y_true = results_dict['predictions']['y_true']
        y_pred = results_dict['predictions']['y_pred']
        
        # Calculate accuracy by count bins
        count_bins = np.arange(0, min(y_true.max(), 20) + 1)
        accuracies = []
        
        for i in range(len(count_bins) - 1):
            mask = (y_true >= count_bins[i]) & (y_true < count_bins[i + 1])
            if np.sum(mask) > 0:
                accuracy = 1 - np.mean(np.abs(y_true[mask] - y_pred[mask]))
                accuracies.append(max(0, accuracy))
            else:
                accuracies.append(0)
        
        ax.bar(count_bins[:-1], accuracies, alpha=0.7, color='skyblue')
        ax.set_xlabel('Count Value')
        ax.set_ylabel('Prediction Accuracy')
        ax.set_title('Accuracy by Count Value')
        ax.grid(True, alpha=0.3)
    
    # 3. Training progress (if available)
    ax = axes[1, 0]
    
    if 'training_history' in results_dict:
        history = results_dict['training_history']
        epochs = range(1, len(history) + 1)
        ax.plot(epochs, history, linewidth=2, color='blue')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Training history not available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Training Progress')
    
    # 4. Summary statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    if 'metrics' in results_dict:
        table_data = []
        for key, value in results_dict['metrics'].items():
            if isinstance(value, (int, float)):
                table_data.append([key.upper(), f'{value:.4f}'])
        
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax.set_title('Model Performance Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_summary.png'), 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def create_advanced_academic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_prob: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    dpi: int = 300
):
    """
    Create additional advanced plots inspired by academic zero-inflated model research.
    """
    
    print("   🔬 Creating advanced academic-style plots...")
    
    # 1. Zero-Inflation Probability Density Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Zero-Inflation Model Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Probability density comparison
    ax = axes[0, 0]
    zero_mask = (y_true == 0)
    nonzero_mask = (y_true > 0)
    
    ax.hist(zero_prob[zero_mask], bins=50, alpha=0.7, label='True Zeros', density=True, color='red')
    ax.hist(zero_prob[nonzero_mask], bins=50, alpha=0.7, label='True Non-zeros', density=True, color='blue')
    ax.set_xlabel('Predicted Zero Probability')
    ax.set_ylabel('Density')
    ax.set_title('Zero Probability Distribution by True Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Quantile-Quantile plot for count predictions
    ax = axes[0, 1]
    count_mask = y_true > 0
    if np.sum(count_mask) > 0:
        y_count_true = y_true[count_mask]
        y_count_pred = y_pred[count_mask]
        
        # Calculate quantiles
        quantiles = np.linspace(0.01, 0.99, 50)
        true_quantiles = np.quantile(y_count_true, quantiles)
        pred_quantiles = np.quantile(y_count_pred, quantiles)
        
        ax.scatter(true_quantiles, pred_quantiles, alpha=0.7, s=30)
        
        # Perfect prediction line
        min_val = min(true_quantiles.min(), pred_quantiles.min())
        max_val = max(true_quantiles.max(), pred_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('True Count Quantiles')
        ax.set_ylabel('Predicted Count Quantiles')
        ax.set_title('Q-Q Plot: Count Model Performance')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Overdispersion Analysis
    ax = axes[1, 0]
    # Calculate observed vs predicted variance for different count ranges
    max_count = min(y_true.max(), 20)
    count_ranges = np.arange(0, max_count + 1)
    obs_var = []
    pred_var = []
    count_means = []
    
    for count in count_ranges:
        mask = (y_true == count)
        if np.sum(mask) > 1:
            obs_var.append(np.var(y_true[mask]))
            pred_var.append(np.var(y_pred[mask]))
            count_means.append(count)
    
    if count_means:
        ax.scatter(count_means, obs_var, label='Observed Variance', alpha=0.7, s=50, color='red')
        ax.scatter(count_means, pred_var, label='Predicted Variance', alpha=0.7, s=50, color='blue')
        
        # Poisson reference line (variance = mean)
        ax.plot(count_means, count_means, 'g--', 
                label='Poisson (Var=Mean)', alpha=0.8)
        
        ax.set_xlabel('Count Value')
        ax.set_ylabel('Variance')
        ax.set_title('Variance vs Mean Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Conditional Mean Analysis
    ax = axes[1, 1]
    # Binned residual analysis
    y_pred_sorted_idx = np.argsort(y_pred)
    n_bins = 20
    bin_size = len(y_pred) // n_bins
    
    bin_means = []
    bin_residuals = []
    bin_centers = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_pred)
        
        bin_idx = y_pred_sorted_idx[start_idx:end_idx]
        bin_true = y_true[bin_idx]
        bin_pred = y_pred[bin_idx]
        
        bin_centers.append(np.mean(bin_pred))
        bin_means.append(np.mean(bin_true))
        bin_residuals.append(np.mean(bin_true - bin_pred))
    
    ax.scatter(bin_centers, bin_residuals, s=80, alpha=0.8, color='red')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.6)
    ax.set_xlabel('Binned Predicted Values')
    ax.set_ylabel('Mean Residuals')
    ax.set_title('Binned Residual Analysis')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'advanced_academic_analysis.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Model Fit Diagnostics (Inspired by academic papers)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Zero-Inflated Model Fit Diagnostics', fontsize=16, fontweight='bold')
    
    # Worm plot (detrended Q-Q plot)
    ax = axes[0, 0]
    residuals = y_true - y_pred
    standardized_residuals = residuals / np.std(residuals)
    
    # Calculate theoretical quantiles
    n = len(standardized_residuals)
    theoretical_quantiles = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    empirical_quantiles = np.sort(standardized_residuals)
    
    # Detrended Q-Q (worm plot)
    detrended = empirical_quantiles - theoretical_quantiles
    ax.scatter(theoretical_quantiles, detrended, alpha=0.6, s=20)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Empirical - Theoretical')
    ax.set_title('Worm Plot (Detrended Q-Q)')
    ax.grid(True, alpha=0.3)
    
    # Zero-inflation pattern visualization
    ax = axes[0, 1]
    zero_counts = np.bincount((y_true == 0).astype(int))
    pred_zero_counts = np.bincount((y_pred < 0.5).astype(int))
    
    categories = ['Non-Zero', 'Zero']
    x_pos = np.arange(len(categories))
    
    width = 0.35
    ax.bar(x_pos - width/2, [len(y_true) - zero_counts[1], zero_counts[1]], 
           width, label='Observed', alpha=0.8, color='skyblue')
    ax.bar(x_pos + width/2, [len(y_pred) - pred_zero_counts[1], pred_zero_counts[1]], 
           width, label='Predicted', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Count Type')
    ax.set_ylabel('Frequency')
    ax.set_title('Zero vs Non-Zero Frequency Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Dispersion Index Plot
    ax = axes[0, 2]
    # Calculate dispersion index for different subgroups
    unique_counts = np.unique(y_true[y_true <= 10])  # Focus on lower counts
    dispersion_indices = []
    sample_sizes = []
    
    for count in unique_counts:
        mask = (y_true == count)
        if np.sum(mask) > 5:  # Need sufficient sample size
            count_pred_subset = y_pred[mask]
            if len(count_pred_subset) > 1:
                mean_pred = np.mean(count_pred_subset)
                var_pred = np.var(count_pred_subset)
                if mean_pred > 0:
                    dispersion_idx = var_pred / mean_pred
                    dispersion_indices.append(dispersion_idx)
                    sample_sizes.append(len(count_pred_subset))
                else:
                    dispersion_indices.append(0)
                    sample_sizes.append(len(count_pred_subset))
    
    if dispersion_indices:
        ax.scatter(unique_counts[:len(dispersion_indices)], dispersion_indices, 
                  s=np.array(sample_sizes)*2, alpha=0.7, c=sample_sizes, cmap='viridis')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.8, label='Equidispersion')
        ax.set_xlabel('True Count Value')
        ax.set_ylabel('Dispersion Index (Var/Mean)')
        ax.set_title('Dispersion Index by Count Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Residuals vs Fitted with density
    ax = axes[1, 1]
    
    residuals = y_true - y_pred
    
    # Create scatter plot with density coloring
    scatter = ax.scatter(y_pred, residuals, c=y_pred, cmap='viridis', alpha=0.6, s=20)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    
    # Add LOWESS trend line
    try:
        from scipy.signal import savgol_filter
        sorted_idx = np.argsort(y_pred)
        if len(y_pred) > 50:
            window_length = min(51, len(y_pred) // 10)
            if window_length % 2 == 0:
                window_length += 1
            smoothed = savgol_filter(residuals[sorted_idx], window_length, 3)
            ax.plot(y_pred[sorted_idx], smoothed, 'r-', linewidth=2, alpha=0.8, label='Trend')
            ax.legend()
    except:
        pass
    
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted Values')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label='Fitted Value')
    
    # 6. Count Model Performance (Non-zeros only)
    ax = axes[1, 2]
    
    nonzero_mask = y_true > 0
    if np.sum(nonzero_mask) > 0:
        y_nonzero_true = y_true[nonzero_mask]
        y_nonzero_pred = y_pred[nonzero_mask]
        
        # Scatter plot with perfect prediction line
        ax.scatter(y_nonzero_true, y_nonzero_pred, alpha=0.6, s=30, color='blue')
        
        # Perfect prediction line
        min_val = min(y_nonzero_true.min(), y_nonzero_pred.min())
        max_val = max(y_nonzero_true.max(), y_nonzero_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        # Calculate R²
        r2_count = r2_score(y_nonzero_true, y_nonzero_pred)
        
        ax.set_xlabel('True Count (Non-zeros)')
        ax.set_ylabel('Predicted Count')
        ax.set_title(f'Count Model Performance (R² = {r2_count:.3f})')
        ax.grid(True, alpha=0.3)
        
        # Add correlation info
        correlation = np.corrcoef(y_nonzero_true, y_nonzero_pred)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 7. Model Performance by Count Range
    ax = axes[2, :]

    # Binned performance analysis
    max_count_analysis = min(y_true.max(), 15)
    count_ranges = np.arange(0, max_count_analysis + 1)
    
    mae_by_count = []
    mse_by_count = []
    sample_sizes = []
    
    for count in count_ranges:
        mask = (y_true == count)
        if np.sum(mask) > 0:
            mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            mse = np.mean((y_true[mask] - y_pred[mask])**2)
            
            mae_by_count.append(mae)
            mse_by_count.append(mse)
            sample_sizes.append(np.sum(mask))
        else:
            mae_by_count.append(0)
            mse_by_count.append(0)
            sample_sizes.append(0)
    
    # Create twin axis
    ax_twin = ax.twinx()
    
    # Plot performance metrics
    line1 = ax.plot(count_ranges, mae_by_count, 'o-', linewidth=2, markersize=6, 
                     color='blue', label='MAE')
    line2 = ax.plot(count_ranges, mse_by_count, 's-', linewidth=2, markersize=6, 
                     color='red', label='MSE')
    
    # Plot sample sizes as bars
    bars = ax_twin.bar(count_ranges, sample_sizes, alpha=0.3, color='gray', 
                        label='Sample Size')
    
    ax.set_xlabel('True Count Value')
    ax.set_ylabel('Error Metrics')
    ax_twin.set_ylabel('Sample Size')
    ax.set_title('Model Performance by Count Value')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    # 8. Feature Importance (if available)
    ax = axes[3, :2]
    
    # This would need to be passed from the model
    # For now, create a placeholder
    ax.text(0.5, 0.5, 'Feature Importance Analysis\n(Requires model-specific implementation)', 
             ha='center', va='center', transform=ax.transAxes, fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    ax.set_title('Feature Importance Analysis')
    
    # 9. Model Summary Statistics
    ax = axes[3, 2]
    
    # Calculate comprehensive statistics
    stats_text = []
    stats_text.append("MODEL SUMMARY STATISTICS")
    stats_text.append("=" * 25)
    stats_text.append(f"Sample Size: {len(y_true):,}")
    stats_text.append(f"Zero Rate: {np.mean(y_true == 0):.1%}")
    stats_text.append(f"Mean Count: {np.mean(y_true):.2f}")
    stats_text.append(f"Max Count: {y_true.max()}")
    stats_text.append("")
    
    # Calculate basic performance metrics from available data
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    zero_accuracy = np.mean((zero_prob > 0.5) == (y_true == 0))
    
    stats_text.append("BASIC METRICS")
    stats_text.append("-" * 20)
    stats_text.append(f"Overall RMSE: {rmse:.3f}")
    stats_text.append(f"Overall MAE: {mae:.3f}")
    stats_text.append(f"Zero Accuracy: {zero_accuracy:.3f}")
    
    ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig(os.path.join(output_dir, 'model_evaluation_summary.png'), 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def create_zero_inflated_research_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_prob: np.ndarray,
    output_dir: str,
    dpi: int = 300
):
    """
    Create research-grade plots specifically for zero-inflated models.
    Inspired by academic literature on zero-inflated count models.
    """
    
    print("   🔬 Creating zero-inflated research plots...")
    
    # Figure 1: Comprehensive Zero-Inflation Analysis
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Zero-Inflated Count Model: Research-Grade Analysis', fontsize=18, fontweight='bold')
    
    y_zero_true = (y_true == 0).astype(int)
    
    # 1. Probability-Probability plot
    ax = axes[0, 0]
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_zero_true, zero_prob, n_bins=20
    )
    
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
            linewidth=2, markersize=6, color='blue', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Perfect Calibration')
    
    # Calculate Brier score
    brier_score = np.mean((zero_prob - y_zero_true) ** 2)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Probability-Probability Plot\n(Brier Score: {brier_score:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Zero-inflation Rate Stability
    ax = axes[0, 1]
    
    # Split data into temporal/sequential chunks to test stability
    n_chunks = 10
    chunk_size = len(y_true) // n_chunks
    chunk_zero_rates_obs = []
    chunk_zero_rates_pred = []
    chunk_indices = []
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else len(y_true)
        
        chunk_y_true = y_true[start_idx:end_idx]
        chunk_zero_prob = zero_prob[start_idx:end_idx]
        
        obs_rate = np.mean(chunk_y_true == 0)
        pred_rate = np.mean(chunk_zero_prob)
        
        chunk_zero_rates_obs.append(obs_rate)
        chunk_zero_rates_pred.append(pred_rate)
        chunk_indices.append(i + 1)
    
    ax.plot(chunk_indices, chunk_zero_rates_obs, 'o-', 
           linewidth=2, markersize=8, color='blue', label='Observed')
    ax.plot(chunk_indices, chunk_zero_rates_pred, 's-', 
           linewidth=2, markersize=8, color='red', label='Predicted')
    
    ax.set_xlabel('Data Chunk')
    ax.set_ylabel('Zero Rate')
    ax.set_title('Zero-Inflation Rate Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Excess Zero Analysis
    ax = axes[0, 2]
    
    # Calculate expected zeros under Poisson vs observed
    nonzero_counts = y_true[y_true > 0]
    if len(nonzero_counts) > 0:
        poisson_lambda = np.mean(nonzero_counts)
        
        # Expected zero probability under Poisson
        expected_zero_prob_poisson = np.exp(-poisson_lambda)
        observed_zero_prob = np.mean(y_true == 0)
        predicted_zero_prob = np.mean(zero_prob)
        
        categories = ['Poisson\nExpected', 'Observed', 'Model\nPredicted']
        zero_rates = [expected_zero_prob_poisson, observed_zero_prob, predicted_zero_prob]
        colors = ['lightblue', 'orange', 'lightgreen']
        
        bars = ax.bar(categories, zero_rates, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, rate in zip(bars, zero_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Zero Probability')
        ax.set_title('Zero-Inflation Comparison')
        ax.grid(True, alpha=0.3)
        
        # Calculate excess zeros
        excess_zeros = observed_zero_prob - expected_zero_prob_poisson
        ax.text(0.5, 0.8, f'Excess Zeros: {excess_zeros:.3f}', 
               transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 4. Vuong Test Approximation
    ax = axes[1, 0]
    
    # Simplified Vuong test for model comparison
    # Compare zero-inflated vs Poisson model
    if len(nonzero_counts) > 0:
        # Log-likelihood for zero-inflated model (approximation)
        zi_ll = (y_zero_true * np.log(zero_prob + 1e-10) + 
                (1 - y_zero_true) * np.log(1 - zero_prob + 1e-10))
        
        # Log-likelihood for Poisson model
        lambda_est = np.mean(y_true)
        poisson_ll = y_true * np.log(lambda_est + 1e-10) - lambda_est
        
        # Vuong statistic (simplified)
        lr_diff = zi_ll - poisson_ll
        vuong_stat = np.sqrt(len(y_true)) * np.mean(lr_diff) / np.std(lr_diff)
        
        ax.hist(lr_diff, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=np.mean(lr_diff), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(lr_diff):.4f}')
        
        ax.set_xlabel('Log-Likelihood Difference (ZI - Poisson)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Vuong Test Approximation\n(Statistic: {vuong_stat:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Rootogram (Academic Style)
    ax = axes[1, 1]
    
    max_count_root = min(max(y_true.max(), y_pred.max()), 12)
    counts_obs = np.bincount(y_true.astype(int), minlength=max_count_root+1)[:max_count_root+1]
    counts_pred = np.bincount(np.round(y_pred).astype(int), minlength=max_count_root+1)[:max_count_root+1]
    
    x_vals = np.arange(len(counts_obs))
    sqrt_obs = np.sqrt(counts_obs)
    sqrt_pred = np.sqrt(counts_pred)
    
    # Traditional rootogram
    ax.bar(x_vals, sqrt_pred, alpha=0.7, color='lightblue', 
          label='Expected (√scale)', edgecolor='blue')
    
    # Hanging bars for observed
    for i, (obs, pred) in enumerate(zip(sqrt_obs, sqrt_pred)):
        if obs > pred:
            ax.bar(i, obs - pred, bottom=pred, alpha=0.8, 
                  color='red', edgecolor='darkred')
        else:
            ax.bar(i, pred - obs, bottom=obs, alpha=0.8, 
                  color='white', edgecolor='red', hatch='///')
    
    ax.set_xlabel('Count Value')
    ax.set_ylabel('Square Root of Frequency')
    ax.set_title('Rootogram (Hanging Histogram)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Overdispersion Index Plot
    ax = axes[1, 2]
    
    # Calculate overdispersion for different count values
    unique_counts = np.unique(y_true[y_true <= 10])
    overdispersion_indices = []
    count_values = []
    
    for count in unique_counts:
        mask = (y_true == count)
        if np.sum(mask) > 10:  # Need sufficient sample size
            subset_pred = y_pred[mask]
            if len(subset_pred) > 1:
                mean_val = np.mean(subset_pred)
                var_val = np.var(subset_pred)
                if mean_val > 0:
                    overdispersion = var_val / mean_val
                    overdispersion_indices.append(overdispersion)
                    count_values.append(count)
    
    if count_values:
        ax.scatter(count_values, overdispersion_indices, s=100, alpha=0.8, 
                  color='red', edgecolor='black', linewidth=2)
        ax.axhline(y=1, color='blue', linestyle='--', linewidth=2, 
                  alpha=0.8, label='Equidispersion')
        
        # Add trend line
        if len(count_values) > 2:
            z = np.polyfit(count_values, overdispersion_indices, 1)
            p = np.poly1d(z)
            ax.plot(count_values, p(count_values), "g-", alpha=0.8, linewidth=2, 
                   label=f'Trend (slope: {z[0]:.3f})')
        
        ax.set_xlabel('Count Value')
        ax.set_ylabel('Overdispersion Index (Var/Mean)')
        ax.set_title('Overdispersion Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 7. Probability Integral Transform (PIT)
    ax = axes[2, 0]
    
    # PIT for zero-inflation component
    pit_values = []
    for i in range(len(y_true)):
        if y_true[i] == 0:
            pit_val = zero_prob[i]
        else:
            # For non-zeros, use cumulative probability
            # Simplified: assume Poisson for non-zero part
            lambda_val = max(y_pred[i], 1e-6)
            pit_val = zero_prob[i] + (1 - zero_prob[i]) * stats.poisson.cdf(y_true[i], lambda_val)
        pit_values.append(pit_val)
    
    pit_values = np.array(pit_values)
    
    # PIT histogram should be uniform if model is well-calibrated
    n, bins, patches = ax.hist(pit_values, bins=20, density=True, alpha=0.7, 
                              color='lightgreen', edgecolor='black')
    
    # Expected uniform density
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, 
              alpha=0.8, label='Uniform (Perfect)')
    
    ax.set_xlabel('PIT Values')
    ax.set_ylabel('Density')
    ax.set_title('Probability Integral Transform')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # KS test for uniformity
    from scipy.stats import kstest
    ks_stat, p_value = kstest(pit_values, 'uniform')
    ax.text(0.05, 0.95, f'KS test: p={p_value:.4f}', 
           transform=ax.transAxes, fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 8. Residual Autocorrelation
    ax = axes[2, 1]
    
    residuals = y_true - y_pred
    max_lag = min(50, len(residuals) // 10)
    
    autocorr = []
    lags = range(1, max_lag + 1)
    
    for lag in lags:
        if lag < len(residuals):
            corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
            autocorr.append(corr if not np.isnan(corr) else 0)
        else:
            autocorr.append(0)
    
    ax.plot(lags, autocorr, 'o-', linewidth=2, markersize=4, color='blue')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Confidence bounds (approximate)
    conf_bound = 1.96 / np.sqrt(len(residuals))
    ax.axhline(y=conf_bound, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=-conf_bound, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Residual Autocorrelation')
    ax.grid(True, alpha=0.3)
    
    # 9. Model Diagnostics Summary
    ax = axes[2, 2]
    
    # Calculate various diagnostic statistics
    diagnostics = {}
    
    # Zero-inflation diagnostics
    diagnostics['Zero Rate (Obs)'] = f"{np.mean(y_true == 0):.3f}"
    diagnostics['Zero Rate (Pred)'] = f"{np.mean(zero_prob):.3f}"
    
    # Count model diagnostics
    nonzero_mask = y_true > 0
    if np.sum(nonzero_mask) > 0:
        count_rmse = np.sqrt(np.mean((y_true[nonzero_mask] - y_pred[nonzero_mask])**2))
        count_mae = np.mean(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]))
        diagnostics['Count RMSE'] = f"{count_rmse:.3f}"
        diagnostics['Count MAE'] = f"{count_mae:.3f}"
    
    # Overall diagnostics
    overall_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    overall_mae = np.mean(np.abs(y_true - y_pred))
    diagnostics['Overall RMSE'] = f"{overall_rmse:.3f}"
    diagnostics['Overall MAE'] = f"{overall_mae:.3f}"
    
    # Format as table
    diag_text = "MODEL DIAGNOSTICS\n" + "="*20 + "\n"
    for key, value in diagnostics.items():
        diag_text += f"{key:<15}: {value}\n"
    
    ax.text(0.05, 0.95, diag_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zero_inflated_research_analysis.png'), 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def create_hurdle_vs_zi_comparison_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_prob: np.ndarray,
    output_dir: str,
    dpi: int = 300
):
    """
    Create plots comparing zero-inflated vs hurdle model concepts.
    """
    
    print("   📊 Creating hurdle vs zero-inflated comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Zero-Inflated vs Hurdle Model Analysis', fontsize=16, fontweight='bold')
    
    y_zero_true = (y_true == 0).astype(int)
    
    # 1. Zero-generation process visualization
    ax = axes[0, 0]
    
    zero_mask = (y_true == 0)
    
    # Structural vs sampling zeros (conceptual)
    # In ZI models, zeros can come from both processes
    # In hurdle models, zeros come only from the hurdle
    
    # Probability density for zeros vs non-zeros
    ax.hist(zero_prob[zero_mask], bins=30, alpha=0.7, 
           label='True Zeros', density=True, color='red')
    ax.hist(zero_prob[~zero_mask], bins=30, alpha=0.7, 
           label='True Non-zeros', density=True, color='blue')
    
    ax.set_xlabel('Predicted Zero Probability')
    ax.set_ylabel('Density')
    ax.set_title('Zero Probability by True Class\n(ZI Model Interpretation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Count distribution comparison
    ax = axes[0, 1]
    
    # ZI model: zeros from mixture, positives from count model
    # Hurdle model: zeros from binary, positives from truncated count
    
    counts = np.arange(0, min(15, y_true.max() + 1))
    obs_freq = np.array([np.mean(y_true == c) for c in counts])
    pred_freq = np.array([np.mean(np.round(y_pred) == c) for c in counts])
    
    x_pos = np.arange(len(counts))
    width = 0.35
    
    ax.bar(x_pos - width/2, obs_freq, width, label='Observed', 
           alpha=0.8, color='skyblue')
    ax.bar(x_pos + width/2, pred_freq, width, label='ZI Model', 
           alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Count Value')
    ax.set_ylabel('Probability')
    ax.set_title('Count Distribution Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(counts)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Excess zero identification
    ax = axes[0, 2]
    
    # Compare predicted vs expected zeros under different assumptions
    nonzero_counts = y_true[y_true > 0]
    if len(nonzero_counts) > 0:
        # Poisson assumption
        lambda_poisson = np.mean(y_true)
        expected_zeros_poisson = np.exp(-lambda_poisson)
        
        # Negative binomial assumption (if applicable)
        if len(nonzero_counts) > 1:
            mean_nb = np.mean(nonzero_counts)
            var_nb = np.var(nonzero_counts)
            if var_nb > mean_nb and mean_nb > 0:
                # Method of moments for negative binomial
                p_nb = mean_nb / var_nb
                r_nb = mean_nb * p_nb / (1 - p_nb)
                expected_zeros_nb = (p_nb) ** r_nb
            else:
                expected_zeros_nb = expected_zeros_poisson
        else:
            expected_zeros_nb = expected_zeros_poisson
        
        observed_zeros = np.mean(y_true == 0)
        predicted_zeros = np.mean(zero_prob)
        
        models = ['Poisson', 'Neg. Binomial', 'Observed', 'ZI Model']
        zero_rates = [expected_zeros_poisson, expected_zeros_nb, 
                     observed_zeros, predicted_zeros]
        colors = ['lightblue', 'lightgreen', 'orange', 'red']
        
        bars = ax.bar(models, zero_rates, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, rate in zip(bars, zero_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Zero Rate')
        ax.set_title('Zero Rate Comparison Across Models')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # 4. Conditional mean analysis
    ax = axes[1, 0]
    
    # E[Y|Y>0] analysis - important for hurdle models
    nonzero_mask = y_true > 0
    if np.sum(nonzero_mask) > 0:
        y_nonzero_true = y_true[nonzero_mask]
        y_nonzero_pred = y_pred[nonzero_mask]
        zero_prob_nonzero = zero_prob[nonzero_mask]
        
        # Bin by zero probability and calculate conditional means
        prob_bins = np.percentile(zero_prob_nonzero, [0, 25, 50, 75, 100])
        bin_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        
        conditional_means_obs = []
        conditional_means_pred = []
        
        for i in range(len(prob_bins) - 1):
            if i == len(prob_bins) - 2:
                mask = (zero_prob_nonzero >= prob_bins[i]) & (zero_prob_nonzero <= prob_bins[i + 1])
            else:
                mask = (zero_prob_nonzero >= prob_bins[i]) & (zero_prob_nonzero < prob_bins[i + 1])
            
            if np.sum(mask) > 0:
                conditional_means_obs.append(np.mean(y_nonzero_true[mask]))
                conditional_means_pred.append(np.mean(y_nonzero_pred[mask]))
            else:
                conditional_means_obs.append(0)
                conditional_means_pred.append(0)
        
        x_pos = np.arange(len(bin_labels))
        width = 0.35
        
        ax.bar(x_pos - width/2, conditional_means_obs, width, 
               label='Observed E[Y|Y>0]', alpha=0.8, color='skyblue')
        ax.bar(x_pos + width/2, conditional_means_pred, width, 
               label='Predicted E[Y|Y>0]', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Zero Probability Quartiles')
        ax.set_ylabel('Conditional Mean')
        ax.set_title('E[Y|Y>0] by Zero Probability')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Model selection criteria approximation
    ax = axes[1, 1]
    
    # Approximate AIC/BIC for model comparison
    n = len(y_true)
    
    # Log-likelihood for zero-inflated model (approximation)
    ll_zi = np.sum(y_zero_true * np.log(zero_prob + 1e-10) + 
                   (1 - y_zero_true) * np.log(1 - zero_prob + 1e-10))
    
    # Add count component
    nonzero_mask = y_true > 0
    if np.sum(nonzero_mask) > 0:
        lambda_est = np.mean(y_pred[nonzero_mask])
        ll_zi += np.sum(stats.poisson.logpmf(y_true[nonzero_mask], lambda_est))
    
    # Parameters: assume 2 for binary + 1 for count model
    k_zi = 3
    
    # Simple Poisson model
    lambda_poisson = np.mean(y_true)
    ll_poisson = np.sum(stats.poisson.logpmf(y_true, lambda_poisson))
    k_poisson = 1
    
    # Calculate information criteria
    aic_zi = 2 * k_zi - 2 * ll_zi
    bic_zi = k_zi * np.log(n) - 2 * ll_zi
    
    aic_poisson = 2 * k_poisson - 2 * ll_poisson
    bic_poisson = k_poisson * np.log(n) - 2 * ll_poisson
    
    criteria = ['AIC', 'BIC']
    zi_scores = [aic_zi, bic_zi]
    poisson_scores = [aic_poisson, bic_poisson]
    
    x_pos = np.arange(len(criteria))
    width = 0.35
    
    ax.bar(x_pos - width/2, zi_scores, width, label='ZI Model', 
           alpha=0.8, color='red')
    ax.bar(x_pos + width/2, poisson_scores, width, label='Poisson', 
           alpha=0.8, color='blue')
    
    ax.set_xlabel('Information Criterion')
    ax.set_ylabel('Score (Lower = Better)')
    ax.set_title('Model Selection Criteria')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(criteria)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Model assumptions check
    ax = axes[1, 2]
    
    # Check key assumptions
    assumptions_text = "MODEL ASSUMPTIONS CHECK\n" + "="*25 + "\n\n"
    
    # Zero-inflation assumption
    zero_rate = np.mean(y_true == 0)
    expected_zero_poisson = np.exp(-np.mean(y_true))
    excess_zeros = zero_rate - expected_zero_poisson
    
    assumptions_text += f"Zero Rate (Obs): {zero_rate:.3f}\n"
    assumptions_text += f"Expected (Poisson): {expected_zero_poisson:.3f}\n"
    assumptions_text += f"Excess Zeros: {excess_zeros:.3f}\n\n"
    
    if excess_zeros > 0.05:  # 5% threshold
        assumptions_text += "✓ Zero-inflation present\n"
    else:
        assumptions_text += "✗ Limited zero-inflation\n"
    
    # Overdispersion check
    if len(nonzero_counts) > 1:
        mean_count = np.mean(nonzero_counts)
        var_count = np.var(nonzero_counts)
        dispersion_ratio = var_count / mean_count if mean_count > 0 else 1
        
        assumptions_text += f"\nOverdispersion Ratio: {dispersion_ratio:.3f}\n"
        if dispersion_ratio > 1.2:
            assumptions_text += "✓ Overdispersion detected\n"
        else:
            assumptions_text += "✗ No significant overdispersion\n"
    
    # Model fit quality
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    assumptions_text += f"\nOverall RMSE: {rmse:.3f}\n"
    
    # Zero classification accuracy
    zero_accuracy = np.mean((zero_prob > 0.5) == (y_true == 0))
    assumptions_text += f"Zero Accuracy: {zero_accuracy:.3f}\n"
    
    ax.text(0.05, 0.95, assumptions_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hurdle_vs_zi_comparison.png'), 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_publication_quality_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_prob: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    results_dict: Dict[str, Any],
    output_dir: str,
    dpi: int = 300
):
    """
    Generate all advanced visualizations for publication-quality analysis.
    """
    
    print("🎨 Generating advanced publication-quality visualizations...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plot sets
    try:
        y_zero_true = (y_true == 0).astype(int)
        
        print("   📊 Creating comprehensive diagnostic plots...")
        create_comprehensive_diagnostic_plots(y_true, y_pred, zero_prob, y_zero_true, output_dir, dpi)
        
        print("   🎯 Creating zero-inflation analysis plots...")
        create_zero_inflation_analysis_plots(y_true, zero_prob, features, feature_names, output_dir, dpi)
        
        print("   📈 Creating count model analysis plots...")
        create_count_model_analysis_plots(y_true, y_pred, output_dir, dpi)
        
        print("   🔍 Creating feature analysis plots...")
        create_feature_analysis_plots(features, feature_names, y_true, zero_prob, output_dir, dpi)
        
        print("   🔬 Creating advanced academic plots...")
        create_advanced_academic_plots(y_true, y_pred, zero_prob, features, feature_names, output_dir, dpi)
        
        print("   📊 Creating zero-inflated research plots...")
        create_zero_inflated_research_plots(y_true, y_pred, zero_prob, output_dir, dpi)
        
        print("   📊 Creating hurdle vs ZI comparison plots...")
        create_hurdle_vs_zi_comparison_plots(y_true, y_pred, zero_prob, output_dir, dpi)
        
        print("   � Creating model evaluation summary...")
        create_model_evaluation_summary(y_true, y_pred, zero_prob, results_dict, output_dir, dpi)
        
        print("   �📋 Creating model summary...")
        create_model_comparison_summary(results_dict, output_dir, dpi)
        
        print("✅ Advanced visualizations complete!")
        
    except Exception as e:
        print(f"⚠️  Warning: Some advanced plots failed to generate: {e}")
        import traceback
        traceback.print_exc()
def create_model_evaluation_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_prob: np.ndarray,
    results_dict: Dict[str, Any],
    output_dir: str,
    dpi: int = 300
):
    """
    Create a comprehensive model evaluation summary with academic-style plots.
    """
    
    print("   📈 Creating model evaluation summary...")
    
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Zero-Inflated Count Model: Comprehensive Evaluation Summary', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Model Performance Metrics Summary
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract key metrics
    metrics_data = []
    metric_names = []
    
    if 'enhanced_metrics' in results_dict:
        em = results_dict['enhanced_metrics']
        metrics_data.extend([
            em.get('zero_auc', 0),
            em.get('zero_f1', 0),
            em.get('overall_r2', 0),
            em.get('count_r2', 0)
        ])
        metric_names.extend(['Zero AUC', 'Zero F1', 'Overall R²', 'Count R²'])
    
    if metrics_data:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax1.bar(range(len(metrics_data)), metrics_data, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels(metric_names, rotation=45, ha='right')
        ax1.set_ylabel('Score')
        ax1.set_title('Key Performance Metrics')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_data):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Confusion Matrix for Zero Classification
    ax2 = fig.add_subplot(gs[0, 1])
    
    y_zero_true = (y_true == 0).astype(int)
    y_zero_pred = (zero_prob > 0.5).astype(int)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_zero_true, y_zero_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax2.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax2.set_title('Zero Classification Confusion Matrix')
    
    # Add text annotations
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontweight='bold')
    
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Non-Zero', 'Zero'])
    ax2.set_yticklabels(['Non-Zero', 'Zero'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Error Distribution Analysis
    ax3 = fig.add_subplot(gs[0, 2])
    
    errors = y_true - y_pred
    
    # Create error histogram with overlaid normal distribution
    n, bins, patches = ax3.hist(errors, bins=50, density=True, alpha=0.7, 
                                color='skyblue', edgecolor='black')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(errors)
    x = np.linspace(errors.min(), errors.max(), 100)
    p = stats.norm.pdf(x, mu, sigma)
    ax3.plot(x, p, 'r-', linewidth=2, label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
    
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.8, label='Perfect Prediction')
    ax3.axvline(x=np.mean(errors), color='red', linestyle=':', alpha=0.8, 
                label=f'Mean Error: {np.mean(errors):.3f}')
    
    ax3.set_xlabel('Prediction Error (True - Predicted)')
    ax3.set_ylabel('Density')
    ax3.set_title('Error Distribution Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Calibration Plot
    ax4 = fig.add_subplot(gs[1, 0])
    
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_zero_true, zero_prob, n_bins=10
    )
    
    ax4.plot(mean_predicted_value, fraction_of_positives, "s-", 
             linewidth=2, markersize=8, label='Model Calibration')
    ax4.plot([0, 1], [0, 1], "k:", alpha=0.8, label="Perfect Calibration")
    
    ax4.set_xlabel('Mean Predicted Probability')
    ax4.set_ylabel('Fraction of Positives')
    ax4.set_title('Calibration Plot (Zero Classification)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Calculate calibration error
    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    ax4.text(0.05, 0.95, f'Calibration Error: {calibration_error:.3f}', 
             transform=ax4.transAxes, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 5. Residuals vs Fitted with density
    ax5 = fig.add_subplot(gs[1, 1])
    
    residuals = y_true - y_pred
    
    # Create scatter plot with density coloring
    scatter = ax5.scatter(y_pred, residuals, c=y_pred, cmap='viridis', alpha=0.6, s=20)
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    
    # Add LOWESS trend line
    try:
        from scipy.signal import savgol_filter
        sorted_idx = np.argsort(y_pred)
        if len(y_pred) > 50:
            window_length = min(51, len(y_pred) // 10)
            if window_length % 2 == 0:
                window_length += 1
            smoothed = savgol_filter(residuals[sorted_idx], window_length, 3)
            ax5.plot(y_pred[sorted_idx], smoothed, 'r-', linewidth=2, alpha=0.8, label='Trend')
            ax5.legend()
    except:
        pass
    
    ax5.set_xlabel('Fitted Values')
    ax5.set_ylabel('Residuals')
    ax5.set_title('Residuals vs Fitted Values')
    ax5.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax5, fraction=0.046, pad=0.04, label='Fitted Value')
    
    # 6. Count Model Performance (Non-zeros only)
    ax6 = fig.add_subplot(gs[1, 2])
    
    nonzero_mask = y_true > 0
    if np.sum(nonzero_mask) > 0:
        y_nonzero_true = y_true[nonzero_mask]
        y_nonzero_pred = y_pred[nonzero_mask]
        
        # Scatter plot with perfect prediction line
        ax6.scatter(y_nonzero_true, y_nonzero_pred, alpha=0.6, s=30, color='blue')
        
        # Perfect prediction line
        min_val = min(y_nonzero_true.min(), y_nonzero_pred.min())
        max_val = max(y_nonzero_true.max(), y_nonzero_pred.max())
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        # Calculate R²
        r2_count = r2_score(y_nonzero_true, y_nonzero_pred)
        
        ax6.set_xlabel('True Count (Non-zeros)')
        ax6.set_ylabel('Predicted Count')
        ax6.set_title(f'Count Model Performance (R² = {r2_count:.3f})')
        ax6.grid(True, alpha=0.3)
        
        # Add correlation info
        correlation = np.corrcoef(y_nonzero_true, y_nonzero_pred)[0, 1]
        ax6.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax6.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 7. Model Performance by Count Range
    ax7 = fig.add_subplot(gs[2, :])
    
    # Binned performance analysis
    max_count_analysis = min(y_true.max(), 15)
    count_ranges = np.arange(0, max_count_analysis + 1)
    
    mae_by_count = []
    mse_by_count = []
    sample_sizes = []
    
    for count in count_ranges:
        mask = (y_true == count)
        if np.sum(mask) > 0:
            mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            mse = np.mean((y_true[mask] - y_pred[mask])**2)
            
            mae_by_count.append(mae)
            mse_by_count.append(mse)
            sample_sizes.append(np.sum(mask))
        else:
            mae_by_count.append(0)
            mse_by_count.append(0)
            sample_sizes.append(0)
    
    # Create twin axis
    ax7_twin = ax7.twinx()
    
    # Plot performance metrics
    line1 = ax7.plot(count_ranges, mae_by_count, 'o-', linewidth=2, markersize=6, 
                     color='blue', label='MAE')
    line2 = ax7.plot(count_ranges, mse_by_count, 's-', linewidth=2, markersize=6, 
                     color='red', label='MSE')
    
    # Plot sample sizes as bars
    bars = ax7_twin.bar(count_ranges, sample_sizes, alpha=0.3, color='gray', 
                        label='Sample Size')
    
    ax7.set_xlabel('True Count Value')
    ax7.set_ylabel('Error Metrics')
    ax7_twin.set_ylabel('Sample Size')
    ax7.set_title('Model Performance by Count Value')
    
    # Combine legends
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_twin.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax7.grid(True, alpha=0.3)
    
    # 8. Feature Importance (if available)
    ax8 = fig.add_subplot(gs[3, :2])
    
    # This would need to be passed from the model
    # For now, create a placeholder
    ax8.text(0.5, 0.5, 'Feature Importance Analysis\n(Requires model-specific implementation)', 
             ha='center', va='center', transform=ax8.transAxes, fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    ax8.set_title('Feature Importance Analysis')
    
    # 9. Model Summary Statistics
    ax9 = fig.add_subplot(gs[3, 2])
    
    # Calculate comprehensive statistics
    stats_text = []
    stats_text.append("MODEL SUMMARY STATISTICS")
    stats_text.append("=" * 25)
    stats_text.append(f"Sample Size: {len(y_true):,}")
    stats_text.append(f"Zero Rate: {np.mean(y_true == 0):.1%}")
    stats_text.append(f"Mean Count: {np.mean(y_true):.2f}")
    stats_text.append(f"Max Count: {y_true.max()}")
    stats_text.append("")
    
    if 'enhanced_metrics' in results_dict:
        em = results_dict['enhanced_metrics']
        stats_text.append("PERFORMANCE METRICS")
        stats_text.append("-" * 20)
        stats_text.append(f"Zero AUC: {em.get('zero_auc', 0):.3f}")
        stats_text.append(f"Zero F1: {em.get('zero_f1', 0):.3f}")
        stats_text.append(f"Overall RMSE: {em.get('overall_rmse', 0):.3f}")
        stats_text.append(f"Overall R²: {em.get('overall_r2', 0):.3f}")
        stats_text.append(f"Count RMSE: {em.get('count_rmse', 0):.3f}")
        stats_text.append(f"Threshold: {em.get('best_threshold', 0.5):.3f}")
    
    ax9.text(0.05, 0.95, '\n'.join(stats_text), transform=ax9.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    plt.savefig(os.path.join(output_dir, 'model_evaluation_summary.png'), 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
