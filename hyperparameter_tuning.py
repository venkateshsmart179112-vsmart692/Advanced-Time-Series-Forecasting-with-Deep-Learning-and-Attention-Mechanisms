"""
Hyperparameter Tuning for Attention Layer
Tests different attention configurations to find optimal setup
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import from previous file
import sys
sys.path.append('.')
from importlib import import_module

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Copy necessary classes
class TimeSeriesDataset:
    """Custom dataset for time series sequences"""
    def __init__(self, data, seq_length=30, pred_length=7):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length, 0]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class SelfAttention(nn.Module):
    """Self-attention mechanism"""
    def __init__(self, hidden_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        output = self.out(attention_output)
        
        return output, attention_weights

class AttentionLSTM(nn.Module):
    """LSTM with Self-Attention"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_length, num_heads=4, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_length = output_length
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = SelfAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_length)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, attention_weights = self.attention(lstm_out)
        combined = self.layer_norm(lstm_out + attn_out)
        last_hidden = combined[:, -1, :]
        
        out = self.relu(self.fc1(last_hidden))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, attention_weights

def prepare_data(df, seq_length=30, pred_length=7, train_ratio=0.8):
    """Prepare data"""
    features = ['target', 'feature1', 'feature2', 'feature3', 'feature4']
    data = df[features].values
    
    n = len(data)
    train_size = int(n * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    train_dataset = TimeSeriesDataset(train_scaled, seq_length, pred_length)
    test_dataset = TimeSeriesDataset(test_scaled, seq_length, pred_length)
    
    return train_dataset, test_dataset, scaler

def train_and_evaluate(model, train_loader, test_loader, epochs, device):
    """Train model and return test loss"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions, _ = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions, _ = model(batch_x)
            test_loss += criterion(predictions, batch_y).item()
    
    return test_loss / len(test_loader)

def hyperparameter_sweep():
    """Perform hyperparameter tuning sweep"""
    print("=" * 60)
    print("STEP 4: Hyperparameter Tuning")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    df = pd.read_csv('timeseries_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Parameter grid
    num_heads_options = [2, 4, 8]
    hidden_dim_options = [64, 128, 256]
    
    results = []
    
    print("\nStarting hyperparameter sweep...")
    print("This will test 9 different configurations")
    
    config_num = 0
    for num_heads in num_heads_options:
        for hidden_dim in hidden_dim_options:
            config_num += 1
            
            # Check if hidden_dim is divisible by num_heads
            if hidden_dim % num_heads != 0:
                print(f"  [{config_num}/9] Skipping: heads={num_heads}, dim={hidden_dim} (not divisible)")
                continue
            
            print(f"  [{config_num}/9] Testing: num_heads={num_heads}, hidden_dim={hidden_dim}")
            
            # Prepare data
            train_dataset, test_dataset, scaler = prepare_data(df)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            model = AttentionLSTM(
                input_dim=5,
                hidden_dim=hidden_dim,
                num_layers=2,
                output_length=7,
                num_heads=num_heads,
                dropout=0.2
            ).to(device)
            
            # Train and evaluate
            test_loss = train_and_evaluate(model, train_loader, test_loader, epochs=30, device=device)
            
            results.append({
                'num_heads': num_heads,
                'hidden_dim': hidden_dim,
                'test_loss': test_loss,
                'rmse': np.sqrt(test_loss)
            })
            
            print(f"       → Test Loss: {test_loss:.6f}, RMSE: {np.sqrt(test_loss):.4f}")
    
    return results

def plot_results(results):
    """Visualize hyperparameter tuning results"""
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Heatmap
    pivot = df_results.pivot(index='num_heads', columns='hidden_dim', values='rmse')
    im = axes[0].imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_xticklabels(pivot.columns)
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_xlabel('Hidden Dimension', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Attention Heads', fontsize=12, fontweight='bold')
    axes[0].set_title('Hyperparameter Tuning: RMSE Heatmap', fontsize=14, fontweight='bold')
    
    # Add values to heatmap
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if not np.isnan(pivot.values[i, j]):
                axes[0].text(j, i, f'{pivot.values[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=axes[0], label='RMSE')
    
    # Plot 2: Bar chart
    df_results_sorted = df_results.sort_values('rmse')
    x_labels = [f"H={row['num_heads']}, D={row['hidden_dim']}" for _, row in df_results_sorted.iterrows()]
    
    bars = axes[1].bar(range(len(df_results_sorted)), df_results_sorted['rmse'], color='steelblue')
    axes[1].set_xticks(range(len(df_results_sorted)))
    axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1].set_xlabel('Configuration (Heads, Dimension)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE', fontsize=12, fontweight='bold')
    axes[1].set_title('Configuration Performance Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Highlight best configuration
    best_idx = df_results_sorted['rmse'].idxmin()
    bars[0].set_color('darkgreen')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Hyperparameter tuning results saved")
    plt.close()

if __name__ == "__main__":
    # Run hyperparameter sweep
    results = hyperparameter_sweep()
    
    # Display results
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 60)
    
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))
    
    # Find best configuration
    best_config = df_results.loc[df_results['rmse'].idxmin()]
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"Number of Heads: {int(best_config['num_heads'])}")
    print(f"Hidden Dimension: {int(best_config['hidden_dim'])}")
    print(f"Test RMSE: {best_config['rmse']:.4f}")
    
    # Save results
    df_results.to_csv('hyperparameter_results.csv', index=False)
    print("\n✓ Results saved to 'hyperparameter_results.csv'")
    
    # Plot results
    plot_results(results)
    
    print("\n" + "=" * 60)
    print("Hyperparameter tuning complete!")
    print("Next: Run '5_visualize_attention.py'")
    print("=" * 60)
