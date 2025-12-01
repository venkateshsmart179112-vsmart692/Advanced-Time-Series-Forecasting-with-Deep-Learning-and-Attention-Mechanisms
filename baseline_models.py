"""
Baseline Models: SARIMA and Prophet
Training traditional forecasting methods for comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def prepare_data(df, train_ratio=0.8):
    """Split data into train and test sets"""
    n = len(df)
    train_size = int(n * train_ratio)
    
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    return train_df, test_df

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional accuracy
    direction_actual = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_actual == direction_pred) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }

def train_sarima(train_df, test_df):
    """Train SARIMA model"""
    print("\nTraining SARIMA model...")
    print("This may take a few minutes...")
    
    # Using SARIMA with weekly seasonality
    # Parameters: (p,d,q) x (P,D,Q,s)
    model = SARIMAX(train_df['target'], 
                    order=(2, 1, 2),  # AR, differencing, MA
                    seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    results = model.fit(disp=False, maxiter=200)
    
    # Forecast
    forecast = results.forecast(steps=len(test_df))
    
    # Calculate metrics
    metrics = calculate_metrics(test_df['target'].values, forecast.values)
    
    print("✓ SARIMA training complete")
    
    return forecast, metrics, results

def train_prophet(train_df, test_df):
    """Train Prophet model"""
    print("\nTraining Prophet model...")
    
    # Prepare data for Prophet
    prophet_train = pd.DataFrame({
        'ds': train_df['date'],
        'y': train_df['target']
    })
    
    # Initialize and fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05
    )
    
    model.fit(prophet_train)
    
    # Create future dataframe
    future = pd.DataFrame({'ds': test_df['date']})
    
    # Forecast
    forecast = model.predict(future)
    predictions = forecast['yhat'].values
    
    # Calculate metrics
    metrics = calculate_metrics(test_df['target'].values, predictions)
    
    print("✓ Prophet training complete")
    
    return predictions, metrics, model

def plot_results(train_df, test_df, sarima_pred, prophet_pred, metrics_sarima, metrics_prophet):
    """Visualize baseline model results"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Predictions comparison
    axes[0].plot(train_df['date'], train_df['target'], label='Training Data', alpha=0.7, linewidth=1)
    axes[0].plot(test_df['date'], test_df['target'], label='Actual Test Data', color='black', linewidth=2)
    axes[0].plot(test_df['date'], sarima_pred, label='SARIMA Predictions', linestyle='--', linewidth=2)
    axes[0].plot(test_df['date'], prophet_pred, label='Prophet Predictions', linestyle='--', linewidth=2)
    axes[0].set_title('Baseline Models: Predictions vs Actual', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Value')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Zoomed test period
    axes[1].plot(test_df['date'], test_df['target'], label='Actual', color='black', linewidth=2, marker='o', markersize=3)
    axes[1].plot(test_df['date'], sarima_pred, label='SARIMA', linestyle='--', linewidth=2, marker='s', markersize=3)
    axes[1].plot(test_df['date'], prophet_pred, label='Prophet', linestyle='--', linewidth=2, marker='^', markersize=3)
    axes[1].set_title('Test Period: Detailed View', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add metrics as text
    metrics_text = f"SARIMA - RMSE: {metrics_sarima['RMSE']:.2f}, MAE: {metrics_sarima['MAE']:.2f}, MAPE: {metrics_sarima['MAPE']:.2f}%\n"
    metrics_text += f"Prophet - RMSE: {metrics_prophet['RMSE']:.2f}, MAE: {metrics_prophet['MAE']:.2f}, MAPE: {metrics_prophet['MAPE']:.2f}%"
    axes[1].text(0.02, 0.98, metrics_text, transform=axes[1].transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('baseline_models_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Results visualization saved as 'baseline_models_results.png'")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: Training Baseline Models (SARIMA & Prophet)")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('timeseries_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\nDataset loaded: {len(df)} samples")
    
    # Split data
    train_df, test_df = prepare_data(df, train_ratio=0.8)
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    # Train SARIMA
    sarima_pred, metrics_sarima, sarima_model = train_sarima(train_df, test_df)
    
    # Train Prophet
    prophet_pred, metrics_prophet, prophet_model = train_prophet(train_df, test_df)
    
    # Display results
    print("\n" + "=" * 60)
    print("BASELINE MODEL PERFORMANCE")
    print("=" * 60)
    
    print("\nSARIMA Metrics:")
    for key, value in metrics_sarima.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nProphet Metrics:")
    for key, value in metrics_prophet.items():
        print(f"  {key}: {value:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Model': ['SARIMA', 'Prophet'],
        'RMSE': [metrics_sarima['RMSE'], metrics_prophet['RMSE']],
        'MAE': [metrics_sarima['MAE'], metrics_prophet['MAE']],
        'MAPE': [metrics_sarima['MAPE'], metrics_prophet['MAPE']],
        'Directional_Accuracy': [metrics_sarima['Directional_Accuracy'], metrics_prophet['Directional_Accuracy']]
    })
    metrics_df.to_csv('baseline_metrics.csv', index=False)
    print("\n✓ Metrics saved to 'baseline_metrics.csv'")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'date': test_df['date'],
        'actual': test_df['target'].values,
        'sarima': sarima_pred.values if hasattr(sarima_pred, 'values') else sarima_pred,
        'prophet': prophet_pred
    })
    predictions_df.to_csv('baseline_predictions.csv', index=False)
    print("✓ Predictions saved to 'baseline_predictions.csv'")
    
    # Plot results
    plot_results(train_df, test_df, sarima_pred, prophet_pred, metrics_sarima, metrics_prophet)
    
    print("\n" + "=" * 60)
    print("Baseline models complete!")
    print("Next: Run '3_attention_model.py'")
    print("=" * 60)
