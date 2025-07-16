import numpy as np
import pandas as pd
from scipy.stats import nbinom, gamma, beta

def generate_zero_inflated_data_enhanced(n_samples=7500000, seed=43):
    """
    Generate enhanced zero-inflated count dataset with more complex patterns.
    
    Parameters:
    - n_samples: int, number of samples to generate
    - seed: int, random seed for reproducibility
    
    Returns:
    - pandas.DataFrame with enhanced patterns and stronger signals
    """
    np.random.seed(seed)
    
    # 1. Generate more diverse covariates
    X1 = np.random.normal(0, 1.5, n_samples)  # Higher variance
    X2 = beta.rvs(2, 5, size=n_samples) * 2 - 1  # Beta distribution for skewness
    X3 = np.random.binomial(1, 0.4, n_samples)  # Higher probability
    
    # Add more complex feature interactions
    X1_squared = X1 ** 2
    X1_X2_interaction = X1 * X2
    X2_log = np.log(np.abs(X2) + 0.1)  # Log transformation
    
    # 2. Enhanced zero-inflation with stronger signals and interactions
    logit_p = (-1.2 + 
               2.5 * X1 +                    # Stronger linear effect
               -1.8 * X2 +                   # Stronger negative effect
               1.2 * X3 +                    # Stronger binary effect
               0.8 * X1_squared +            # Quadratic effect
               -0.6 * X1_X2_interaction +    # Interaction effect
               0.4 * X2_log)                 # Log transformation effect
    
    # Apply sigmoid with temperature scaling for better separation
    pi = 1 / (1 + np.exp(-logit_p * 1.5))
    
    # 3. Enhanced count process with complex patterns
    log_mu = (0.8 + 
              1.4 * X1 +                    # Stronger effect
              0.9 * X2 +                    # Stronger effect
              -0.7 * X3 +                   # Stronger negative effect
              0.5 * X1_squared +            # Quadratic relationship
              0.3 * X1_X2_interaction +     # Interaction
              -0.2 * X2_log)                # Log effect
    
    mu = np.exp(log_mu)
    
    # 4. Variable dispersion based on features
    base_dispersion = 1.5
    dispersion_modifier = 1 + 0.5 * np.abs(X1) + 0.3 * X3
    dispersion = base_dispersion * dispersion_modifier
    
    # 5. Draw counts with variable dispersion
    counts = []
    for i in range(n_samples):
        if i % 100000 == 0:  # Progress indicator
            print(f"Enhanced generation progress: {i/n_samples*100:.1f}%")
        count = nbinom.rvs(dispersion[i], dispersion[i] / (dispersion[i] + mu[i]))
        counts.append(count)
    counts = np.array(counts)
    
    # 6. Apply structural zeros with probability
    is_struct_zero = np.random.rand(n_samples) < pi
    y = counts.copy()
    y[is_struct_zero] = 0
    
    # 7. Add some extreme values (fat tail)
    extreme_mask = np.random.rand(n_samples) < 0.01  # 1% extreme values
    extreme_counts = gamma.rvs(a=2, scale=10, size=np.sum(extreme_mask))
    y[extreme_mask] = extreme_counts.astype(int)
    
    # 8. Assemble enhanced DataFrame
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'y': y
    })
    df['zero'] = (df['y'] == 0).astype(int)
    df['data_source'] = 'enhanced'
    
    return df

def generate_zero_inflated_data(n_samples=20000000, seed=42):
    """
    Generate a purely enhanced zero-inflated count dataset.
    
    Parameters:
    - n_samples: int, total number of samples to generate
    - seed: int, random seed for reproducibility
    
    Returns:
    - pandas.DataFrame with enhanced patterns for model training
    """
    print(f"Generating {n_samples:,} samples using enhanced patterns...")
    
    # Generate the dataset
    df_enhanced = generate_zero_inflated_data_enhanced(n_samples, seed + 1)
    
    # Shuffle the dataset
    print("\nShuffling dataset...")
    df_shuffled = df_enhanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Remove data_source column for consistency
    df_final = df_shuffled.drop('data_source', axis=1)
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"- Total samples: {len(df_final):,}")
    print(f"- Zero proportion: {df_final['zero'].mean():.3f}")
    print(f"- Mean count (non-zero): {df_final[df_final['y'] > 0]['y'].mean():.2f}")
    print(f"- Max count: {df_final['y'].max()}")
    print(f"- 99th percentile: {df_final['y'].quantile(0.99):.1f}")
    
    # Feature correlations with zero indicator
    for col in ['X1', 'X2', 'X3']:
        corr = df_final[col].corr(df_final['zero'])
        print(f"- {col} correlation with zero: {corr:.3f}")
    
    return df_final

if __name__ == "__main__":
    # Generate combined dataset with both basic and enhanced patterns
    print("="*60)
    print("ENHANCED ZERO-INFLATED COUNT DATA GENERATION")
    print("="*60)
    
    # Generate 20M samples with only enhanced patterns
    df = generate_zero_inflated_data(
        n_samples=20000000, 
        seed=42
    )
    
    # Save dataset
    output_filename = "zero_inflated_data.csv"
    print(f"\nSaving dataset to {output_filename}...")
    df.to_csv(output_filename, index=False)
    print(f"✅ Successfully generated and saved {len(df):,} samples!")
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE - Ready for training!")
    print("="*60)