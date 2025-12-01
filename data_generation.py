"""
Advanced Time Series Dataset Generation
Generates a complex multi-variate time series with seasonality, trend, and autorregressive components
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

def generate_complex_timeseries(n_samples=2000, n_features=5):
    """
    Generate a complex multi-variate time series dataset
    
    Features:
    1. Target variable with multiple seasonal patterns
    2. Trend component
    3. Multiple correlated features
    4. Autoregressive behavior
    """
    
    # Time index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Base components
    t = np.arange(n_samples)
    
    # Trend component (long-term growth)
    trend = 0.05 * t + 10
    
    # Multiple seasonal patterns
    # Weekly seasonality (period = 7)
    weekly_season = 3 * np.sin(2 * np.pi * t / 7)
    
    # Monthly seasonality (period = 30)
    monthly_season = 5 * np.sin(2 * np.pi * t / 30)
    
    # Yearly seasonality (period = 365)
    yearly_season = 8 * np.sin(2 * np.pi * t / 365)
    
    # Autoregressive component (AR)
    ar_component = np.zeros(n_samples)
    ar_component[0] = np.random.randn()
    for i in range(1, n_samples):
        ar_component[i] = 0.7 * ar_component[i-1] + np.random.randn() * 0.5
    
    # Target variable
    target = trend + weekly_season + monthly_season + yearly_season + ar_component + np.random.randn(n_samples) * 2
    
    # Generate correlated features
    feature1 = 0.8 * target + np.random.randn(n_samples) * 3 + 5  # Highly correlated
    feature2 = 0.5 * np.sin(2 * np.pi * t / 14) + 0.3 * trend + np.random.randn(n_samples) * 2  # Bi-weekly pattern
    feature3 = 0.6 * target + 0.4 * ar_component + np.random.randn(n_samples) * 2.5  # Mixed correlation
    feature4 = np.cos(2 * np.pi * t / 90) * 4 + np.random.randn(n_samples) * 1.5  # Quarterly pattern
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'target': target,
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4
    })
    
    return df

def plot_dataset(df):
    """Visualize the generated dataset"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Generated Complex Time Series Dataset', fontsize=16)
    
    # Plot target
    axes[0, 0].plot(df['date'], df['target'], linewidth=0.8)
    axes[0, 0].set_title('Target Variable (with trend & seasonality)')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot features
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4']
    for idx, feature in enumerate(feature_names):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        axes[row, col].plot(df['date'], df[feature], linewidth=0.8, color=f'C{idx+1}')
        axes[row, col].set_title(f'{feature.capitalize()}')
        axes[row, col].set_xlabel('Date')
        axes[row, col].set_ylabel('Value')
        axes[row, col].grid(True, alpha=0.3)
    
    # Plot correlation matrix
    axes[2, 1].axis('off')
    corr = df[['target', 'feature1', 'feature2', 'feature3', 'feature4']].corr()
    im = axes[2, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[2, 1].set_xticks(range(5))
    axes[2, 1].set_yticks(range(5))
    axes[2, 1].set_xticklabels(['target', 'f1', 'f2', 'f3', 'f4'])
    axes[2, 1].set_yticklabels(['target', 'f1', 'f2', 'f3', 'f4'])
    axes[2, 1].set_title('Correlation Matrix')
    axes[2, 1].axis('on')
    plt.colorbar(im, ax=axes[2, 1])
    
    for i in range(5):
        for j in range(5):
            axes[2, 1].text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig('dataset_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Dataset visualization saved as 'dataset_visualization.png'")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Generating Complex Time Series Dataset")
    print("=" * 60)
    
    # Generate dataset
    df = generate_complex_timeseries(n_samples=2000, n_features=5)
    
    # Save to CSV
    df.to_csv('timeseries_data.csv', index=False)
    print(f"\n✓ Dataset generated with {len(df)} samples")
    print(f"✓ Saved to 'timeseries_data.csv'")
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(df.describe())
    
    # Plot dataset
    plot_dataset(df)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("Next: Run '2_baseline_models.py'")
    print("=" * 60)
