"""
Advanced feature engineering for improved model performance.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering transformer."""
    
    def __init__(self, 
                 create_polynomial_features=True,
                 polynomial_degree=2,
                 create_log_features=True,
                 create_interaction_features=True,
                 create_binned_features=True,
                 n_bins=10,
                 create_statistical_features=True,
                 random_state=42):
        self.create_polynomial_features = create_polynomial_features
        self.polynomial_degree = polynomial_degree
        self.create_log_features = create_log_features
        self.create_interaction_features = create_interaction_features
        self.create_binned_features = create_binned_features
        self.n_bins = n_bins
        self.create_statistical_features = create_statistical_features
        self.random_state = random_state
        self.feature_names_ = None
        self.bin_edges_ = None
        self.transformed_feature_names_ = None  # Added attribute to store transformed feature names
        
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f'X{i}' for i in range(X.shape[1])]
        
        # Fit binning edges
        if self.create_binned_features:
            self.bin_edges_ = {}
            for i, feature in enumerate(self.feature_names_):
                if hasattr(X, 'iloc'):
                    values = X.iloc[:, i]
                else:
                    values = X[:, i]
                self.bin_edges_[feature] = np.percentile(values, 
                                                       np.linspace(0, 100, self.n_bins + 1))
        
        return self
    
    def transform(self, X):
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
            
        features = [X_array]
        new_feature_names = self.feature_names_.copy()
        
        # 1. Log transformations (for positive skewed features)
        if self.create_log_features:
            log_features = np.log1p(np.maximum(X_array, 0))  # log(1 + max(x, 0))
            features.append(log_features)
            new_feature_names.extend([f'{name}_log' for name in self.feature_names_])
        
        # 2. Polynomial features
        if self.create_polynomial_features and self.polynomial_degree > 1:
            poly_features = []
            for degree in range(2, self.polynomial_degree + 1):
                poly_features.append(X_array ** degree)
                new_feature_names.extend([f'{name}_pow{degree}' for name in self.feature_names_])
            if poly_features:
                features.extend(poly_features)
        
        # 3. Interaction features (pairwise products)
        if self.create_interaction_features and X_array.shape[1] > 1:
            interaction_features = []
            interaction_names = []
            for i in range(X_array.shape[1]):
                for j in range(i + 1, X_array.shape[1]):
                    interaction_features.append((X_array[:, i] * X_array[:, j]).reshape(-1, 1))
                    interaction_names.append(f'{self.feature_names_[i]}_x_{self.feature_names_[j]}')
            
            if interaction_features:
                features.append(np.hstack(interaction_features))
                new_feature_names.extend(interaction_names)
        
        # 4. Binned features (categorical encoding of continuous variables)
        if self.create_binned_features and self.bin_edges_:
            binned_features = []
            for i, feature in enumerate(self.feature_names_):
                if hasattr(X, 'iloc'):
                    values = X.iloc[:, i]
                else:
                    values = X_array[:, i]
                
                binned = np.digitize(values, self.bin_edges_[feature]) - 1
                binned = np.clip(binned, 0, self.n_bins - 1)  # Ensure within bounds
                
                # One-hot encode the bins
                one_hot = np.zeros((len(values), self.n_bins))
                one_hot[np.arange(len(values)), binned] = 1
                binned_features.append(one_hot)
                
            if binned_features:
                features.append(np.hstack(binned_features))
                bin_names = []
                for feature in self.feature_names_:
                    bin_names.extend([f'{feature}_bin_{i}' for i in range(self.n_bins)])
                new_feature_names.extend(bin_names)
        
        # 5. Statistical features (rolling statistics if we treat as time series-like)
        if self.create_statistical_features:
            # Mean and std across features (row-wise statistics)
            row_mean = np.mean(X_array, axis=1, keepdims=True)
            row_std = np.std(X_array, axis=1, keepdims=True)
            row_min = np.min(X_array, axis=1, keepdims=True)
            row_max = np.max(X_array, axis=1, keepdims=True)
            row_range = row_max - row_min;
            
            stat_features = np.hstack([row_mean, row_std, row_min, row_max, row_range])
            features.append(stat_features)
            new_feature_names.extend(['row_mean', 'row_std', 'row_min', 'row_max', 'row_range'])
        
        # Combine all features
        X_transformed = np.hstack(features)
        
        self.transformed_feature_names_ = new_feature_names
        
        return X_transformed

    def get_feature_names(self):
        """Returns the names of the transformed features."""
        if not hasattr(self, 'transformed_feature_names_') or self.transformed_feature_names_ is None:
            raise RuntimeError("Transformer has not been fitted or transformed yet.")
        return self.transformed_feature_names_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class SmartFeatureSelector(BaseEstimator, TransformerMixin):
    """Smart feature selection using multiple criteria."""
    
    def __init__(self, 
                 max_features=100,
                 selection_methods=['mutual_info', 'f_test'],
                 remove_low_variance=True,
                 variance_threshold=0.01,
                 remove_highly_correlated=True,
                 correlation_threshold=0.95):
        self.max_features = max_features
        self.selection_methods = selection_methods
        self.remove_low_variance = remove_low_variance
        self.variance_threshold = variance_threshold
        self.remove_highly_correlated = remove_highly_correlated
        self.correlation_threshold = correlation_threshold
        self.selected_features_ = None
        
    def fit(self, X, y):
        # Start with all features
        n_features = X.shape[1]
        selected_mask = np.ones(n_features, dtype=bool)
        
        # 1. Remove low variance features
        if self.remove_low_variance:
            variances = np.var(X, axis=0)
            low_var_mask = variances > self.variance_threshold
            selected_mask &= low_var_mask
            print(f"Removed {np.sum(~low_var_mask)} low variance features")
        
        # Apply current selection
        X_selected = X[:, selected_mask]
        
        # 2. Remove highly correlated features
        if self.remove_highly_correlated and X_selected.shape[1] > 1:
            corr_matrix = np.corrcoef(X_selected.T)
            corr_matrix = np.abs(corr_matrix)
            
            # Find features to remove
            upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            high_corr_pairs = np.where((corr_matrix > self.correlation_threshold) & upper_tri)
            
            features_to_remove = set()
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                # Keep the feature with higher variance
                if np.var(X_selected[:, i]) > np.var(X_selected[:, j]):
                    features_to_remove.add(j)
                else:
                    features_to_remove.add(i)
            
            # Update mask
            selected_indices = np.where(selected_mask)[0]
            for idx in features_to_remove:
                selected_mask[selected_indices[idx]] = False
            
            print(f"Removed {len(features_to_remove)} highly correlated features")
        
        # Apply current selection
        X_selected = X[:, selected_mask]
        
        # 3. Statistical feature selection
        if len(self.selection_methods) > 0 and X_selected.shape[1] > self.max_features:
            scores = np.zeros(X_selected.shape[1])
            
            for method in self.selection_methods:
                if method == 'mutual_info':
                    if np.unique(y).size == 2:  # Classification
                        method_scores = mutual_info_classif(X_selected, y, random_state=42)
                    else:  # Regression
                        from sklearn.feature_selection import mutual_info_regression
                        method_scores = mutual_info_regression(X_selected, y, random_state=42)
                elif method == 'f_test':
                    if np.unique(y).size == 2:  # Classification
                        method_scores, _ = f_classif(X_selected, y)
                    else:  # Regression
                        method_scores, _ = f_regression(X_selected, y)
                
                # Normalize scores to [0, 1]
                if method_scores.max() > method_scores.min():
                    method_scores = (method_scores - method_scores.min()) / (method_scores.max() - method_scores.min())
                scores += method_scores
            
            # Select top features
            top_indices = np.argsort(scores)[-self.max_features:]
            selected_indices = np.where(selected_mask)[0]
            final_mask = np.zeros(n_features, dtype=bool)
            final_mask[selected_indices[top_indices]] = True
            selected_mask = final_mask
        
        self.selected_features_ = selected_mask
        print(f"Selected {np.sum(selected_mask)} features out of {n_features}")
        
        return self
    
    def transform(self, X):
        return X[:, self.selected_features_]


def create_advanced_feature_pipeline(config, verbose=False):
    """Create advanced feature engineering pipeline."""
    from sklearn.pipeline import Pipeline
    
    steps = [
        ('advanced_features', AdvancedFeatureEngineer(
            create_polynomial_features=True,
            polynomial_degree=3,  # Match config
            create_log_features=True,
            create_interaction_features=True,
            create_binned_features=True,
            n_bins=5,  # Reduce bins to avoid too many features
            create_statistical_features=True,
            random_state=config.general.random_seed
        )),
        ('scaler', RobustScaler())  # More robust to outliers than StandardScaler
        # SmartFeatureSelector REMOVED - Let GPU models handle feature selection
        # Apple M3 Pro can easily handle 50+ features with neural networks
    ]
    
    pipeline = Pipeline(steps)
    
    if verbose:
        print("Created advanced feature engineering pipeline:")
        for step_name, step in steps:
            print(f"  - {step_name}: {step.__class__.__name__}")
        print("  - SmartFeatureSelector: DISABLED (preserving all engineered features)")
    
    return pipeline


class AdvancedFeatureEngineering:
    def __init__(self, poly_degree=2, interaction_only=False):
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=interaction_only)
        self.feature_names = None

    def _clean_names(self, df):
        # LightGBM can have issues with special characters in column names
        df.columns = df.columns.str.replace(' ', '_').str.replace('^', '_pow_')
        return df

    def fit_transform(self, df):
        """
        Engineers advanced features including interactions and custom transformations.
        """
        X_base = df[['X1', 'X2', 'X3']]

        # 1. Generate polynomial features (includes base features, interactions, and powers)
        poly_features = self.poly.fit_transform(X_base)
        poly_names = self.poly.get_feature_names_out(['X1', 'X2', 'X3'])
        features_df = pd.DataFrame(poly_features, columns=poly_names, index=df.index)
        features_df = self._clean_names(features_df)

        # 2. Add specific transformations not captured by polynomial features
        features_df['X2_log'] = np.log(np.abs(df['X2']) + 0.1)
        
        self.feature_names = features_df.columns.tolist()

        # 3. Combine with original target variables
        final_df = pd.concat([features_df, df[['y', 'zero']]], axis=1)
        
        return final_df

    def transform(self, df):
        """
        Applies the same feature engineering to new data.
        """
        X_base = df[['X1', 'X2', 'X3']]

        # 1. Generate polynomial features
        poly_features = self.poly.transform(X_base)
        poly_names = self.poly.get_feature_names_out(['X1', 'X2', 'X3'])
        features_df = pd.DataFrame(poly_features, columns=poly_names, index=df.index)
        features_df = self._clean_names(features_df)

        # 2. Add specific transformations
        features_df['X2_log'] = np.log(np.abs(df['X2']) + 0.1)

        # 3. Ensure columns match the training set
        # Add any missing columns
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder and select to match training
        features_df = features_df[self.feature_names]

        # 4. Combine with original target variables, if they exist
        target_cols = [col for col in ['y', 'zero'] if col in df.columns]
        final_df = pd.concat([features_df, df[target_cols]], axis=1)
        
        return final_df

    def get_feature_names(self):
        return self.feature_names
