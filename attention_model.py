"""
Advanced Attention-Enhanced LSTM Model
Implements sequence-to-sequence forecasting with self-attention mechanism
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series sequences"""
    
    def __init__(self, data, seq_length=30, pred_length=7):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length, 0]  # Predict target only
        return torch.FloatTensor(x), torch.FloatTensor(y)

class SelfAttention(nn.Module):
    """Self-attention mechanism"""
    
    def __init__(self, hidden_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        # Linear projections and reshape for multi-head attention
        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        output = self.out(attention_output)
        
        return output, attention_weights

class AttentionLSTM(nn.Module):
    """LSTM with Self-Attention for Time Series Forecasting"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_length, num_heads=4, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_length = output_length
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Self-attention
        self.attention = SelfAttention(hidden_dim, num_heads)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_length)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, attention_weights = self.attention(lstm_out)
        
        # Residual connection and layer norm
        combined = self.layer_norm(lstm_out + attn_out)
        
        # Take the last time step
        last_hidden = combined[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_hidden))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, attention_weights

def prepare_data(df, seq_length=30, pred_length=7, train_ratio=0.8):
    """Prepare and scale data"""
    # Select features
    features = ['target', 'feature1', 'feature2', 'feature3', 'feature4']
    data = df[features].values
    
    # Split train/test
    n = len(data)
    train_size = int(n * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Scale data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_scaled, seq_length, pred_length)
    test_dataset = TimeSeriesDataset(test_scaled, seq_length, pred_length)
    
    return train_dataset, test_dataset, scaler, train_data, test_data

def train_model(model, train_loader, epochs, device):
    """Train the model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    
    model.train()
    train_losses = []
    
    print("\nTraining Progress:")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return train_losses

def evaluate_model(model, test_loader, scaler, device, pred_length):
    """Evaluate model and extract attention weights"""
    model.eval()
    predictions = []
    actuals = []
    attention_weights_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred, attn_weights = model(batch_x)
            
            predictions.extend(pred.cpu().numpy())
            actuals.extend(batch_y.numpy())
            
            # Store attention weights from last head, last batch
            attention_weights_list.append(attn_weights[0, -1, :, :].cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform predictions (only target column)
    pred_full = np.zeros((predictions.shape[0], predictions.shape[1], 5))
    actual_full = np.zeros((actuals.shape[0], actuals.shape[1], 5))
    
    pred_full[:, :, 0] = predictions
    actual_full[:, :, 0] = actuals
    
    pred_inv = scaler.inverse_transform(pred_full.reshape(-1, 5))[:, 0].reshape(predictions.shape)
    actual_inv = scaler.inverse_transform(actual_full.reshape(-1, 5))[:, 0].reshape(actuals.shape)
    
    return pred_inv, actual_inv, attention_weights_list

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    # Flatten arrays for multi-step predictions
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def plot_training(train_losses):
    """Plot training loss"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, linewidth=2)
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    print("✓ Training loss plot saved")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: Training Attention-Enhanced LSTM Model")
    print("=" * 60)
    
    # Parameters
    SEQ_LENGTH = 30
    PRED_LENGTH = 7
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    NUM_HEADS = 4
    BATCH_SIZE = 32
    EPOCHS = 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load and prepare data
    df = pd.read_csv('timeseries_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    train_dataset, test_dataset, scaler, train_data, test_data = prepare_data(
        df, SEQ_LENGTH, PRED_LENGTH, train_ratio=0.8
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nTraining sequences: {len(train_dataset)}")
    print(f"Testing sequences: {len(test_dataset)}")
    
    # Initialize model
    model = AttentionLSTM(
        input_dim=5,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_length=PRED_LENGTH,
        num_heads=NUM_HEADS
    ).to(device)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses = train_model(model, train_loader, EPOCHS, device)
    print("✓ Training complete")
    
    # Save model
    torch.save(model.state_dict(), 'attention_lstm_model.pth')
    print("✓ Model saved to 'attention_lstm_model.pth'")
    
    # Evaluate
    print("\nEvaluating model...")
    predictions, actuals, attention_weights = evaluate_model(
        model, test_loader, scaler, device, PRED_LENGTH
    )
    
    # Calculate metrics
    metrics = calculate_metrics(actuals, predictions)
    
    print("\n" + "=" * 60)
    print("ATTENTION-LSTM PERFORMANCE")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    np.save('attention_predictions.npy', predictions)
    np.save('attention_actuals.npy', actuals)
    np.save('attention_weights.npy', attention_weights)
    print("\n✓ Predictions and attention weights saved")
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv('attention_metrics.csv', index=False)
    
    # Plot training
    plot_training(train_losses)
    
    print("\n" + "=" * 60)
    print("Attention model training complete!")
    print("Next: Run '4_hyperparameter_tuning.py'")
    print("=" * 60)
