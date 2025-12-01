"""
Attention Weights Visualization and Analysis
Analyzes and visualizes what patterns the attention mechanism learns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def load_attention_data():
    """Load saved attention weights and predictions"""
    attention_weights = np.load('attention_weights.npy', allow_pickle=True)
    predictions = np.load('attention_predictions.npy')
    actuals = np.load('attention_actuals.npy')
    
    return attention_weights, predictions, actuals

def visualize_attention_heatmap(attention_weights, num_samples=5):
    """Create attention heatmap visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Attention Weight Heatmaps: Learning Temporal Dependencies', 
                 fontsize=16, fontweight='bold')
    
    # Select diverse samples
    sample_indices = np.linspace(0, len(attention_weights)-1, num_samples, dtype=int)[:6]
    
    for idx, sample_idx in enumerate(sample_indices):
        row = idx // 3
        col = idx % 3
        
        # Get attention weights for this sample
        attn = attention_weights[sample_idx]
        
        # Plot heatmap
        im = axes[row, col].imshow(attn, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        axes[row, col].set_xlabel('Key Position (Past Time Steps)', fontsize=10)
        axes[row, col].set_ylabel('Query Position (Time Steps)', fontsize=10)
        axes[row, col].set_title(f'Sample {sample_idx + 1}', fontsize=11, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Add grid
        axes[row, col].set_xticks(np.arange(0, attn.shape[1], 5))
        axes[row, col].set_yticks(np.arange(0, attn.shape[0], 5))
        axes[row, col].grid(False)
    
    plt.tight_layout()
    plt.savefig('attention_heatmaps.png', dpi=300, bbox_inches='tight')
    print("✓ Attention heatmaps saved")
    plt.close()

def analyze_temporal_patterns(attention_weights):
    """Analyze which time lags the model focuses on"""
    # Average attention weights across all samples
    avg_attention = np.mean([w.mean(axis=0) for w in attention_weights], axis=0)
    
    # Find top attended positions
    top_k = 10
    top_positions = np.argsort(avg_attention)[-top_k:][::-1]
    top_weights = avg_attention[top_positions]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Average attention over time
    axes[0].plot(avg_attention, linewidth=2, color='darkblue', marker='o', markersize=4)
    axes[0].axhline(y=avg_attention.mean(), color='red', linestyle='--', 
                    label=f'Mean: {avg_attention.mean():.4f}', linewidth=2)
    axes[0].set_xlabel('Time Step (Lag)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Attention Weight', fontsize=12, fontweight='bold')
    axes[0].set_title('Average Attention Distribution Across Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Highlight important regions
    for pos in top_positions[:3]:
        axes[0].axvspan(pos-0.5, pos+0.5, alpha=0.2, color='gold')
    
    # Plot 2: Top attended positions
    axes[1].barh(range(top_k), top_weights, color='steelblue')
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels([f'Lag {pos}' for pos in top_positions])
    axes[1].set_xlabel('Attention Weight', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Top {top_k} Most Attended Time Lags', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (pos, weight) in enumerate(zip(top_positions, top_weights)):
        axes[1].text(weight, i, f' {weight:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('temporal_attention_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Temporal analysis saved")
    plt.close()
    
    return top_positions, top_weights

def create_attention_interpretation(top_positions, top_weights):
    """Generate textual interpretation of attention patterns"""
    interpretation = []
    
    # Categorize attention patterns
    recent_focus = sum([w for p, w in zip(top_positions, top_weights) if p < 7])
    medium_focus = sum([w for p, w in zip(top_positions, top_weights) if 7 <= p < 15])
    distant_focus = sum([w for p, w in zip(top_positions, top_weights) if p >= 15])
    
    interpretation.append("=" * 70)
    interpretation.append("ATTENTION PATTERN INTERPRETATION")
    interpretation.append("=" * 70)
    interpretation.append("")
    
    interpretation.append("1. TEMPORAL FOCUS DISTRIBUTION")
    interpretation.append(f"   • Recent past (1-7 days): {recent_focus:.2%} of top attention")
    interpretation.append(f"   • Medium past (8-15 days): {medium_focus:.2%} of top attention")
    interpretation.append(f"   • Distant past (16+ days): {distant_focus:.2%} of top attention")
    interpretation.append("")
    
    interpretation.append("2. KEY FINDINGS")
    if recent_focus > 0.5:
        interpretation.append("   • The model heavily relies on RECENT observations (past week)")
        interpretation.append("   • This suggests strong short-term dependencies in the data")
    elif medium_focus > 0.5:
        interpretation.append("   • The model focuses on MEDIUM-TERM patterns (1-2 weeks)")
        interpretation.append("   • This indicates bi-weekly or semi-monthly cycles")
    else:
        interpretation.append("   • The model attends to LONG-TERM patterns (2+ weeks)")
        interpretation.append("   • This reveals seasonal or monthly dependencies")
    
    interpretation.append("")
    interpretation.append("3. TOP ATTENDED TIME LAGS")
    for i, (pos, weight) in enumerate(zip(top_positions[:5], top_weights[:5])):
        interpretation.append(f"   {i+1}. Lag {pos}: {weight:.4f} ({weight/top_weights.sum()*100:.1f}% of top-10)")
        
        # Contextual interpretation
        if pos <= 1:
            context = "immediate past (yesterday)"
        elif pos <= 7:
            context = f"recent past (~{pos} days ago, last week)"
        elif pos <= 14:
            context = f"medium past (~{pos} days ago, bi-weekly)"
        elif pos <= 21:
            context = f"three weeks ago"
        else:
            context = f"monthly pattern (~{pos} days ago)"
        interpretation.append(f"      → {context}")
    
    interpretation.append("")
    interpretation.append("4. PRACTICAL INSIGHTS")
    interpretation.append("   The attention mechanism has learned to:")
    
    if top_positions[0] < 7:
        interpretation.append("   • Prioritize recent trends for short-term forecasting")
    if any(7 <= p <= 14 for p in top_positions[:5]):
        interpretation.append("   • Capture weekly and bi-weekly patterns")
    if any(p >= 20 for p in top_positions[:5]):
        interpretation.append("   • Consider long-term seasonal effects")
    
    interpretation.append("")
    interpretation.append("   This selective attention allows the model to:")
    interpretation.append("   • Focus on the MOST RELEVANT historical data")
    interpretation.append("   • Ignore noise and less important time periods")
    interpretation.append("   • Adapt predictions based on temporal context")
    interpretation.append("")
    interpretation.append("=" * 70)
    
    return "\n".join(interpretation)

def compare_predictions(predictions, actuals):
    """Visualize prediction quality"""
    # Take first prediction from each sample (1-step ahead)
    pred_1step = predictions[:, 0]
    actual_1step = actuals[:, 0]
    
    # Take last prediction from each sample (7-step ahead)
    pred_7step = predictions[:, -1]
    actual_7step = actuals[:, -1]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: 1-step ahead predictions
    axes[0, 0].plot(actual_1step[:100], label='Actual', linewidth=2, marker='o', markersize=3)
    axes[0, 0].plot(pred_1step[:100], label='Predicted', linewidth=2, marker='s', markersize=3, alpha=0.7)
    axes[0, 0].set_title('1-Step Ahead Predictions (First 100 samples)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: 7-step ahead predictions
    axes[0, 1].plot(actual_7step[:100], label='Actual', linewidth=2, marker='o', markersize=3)
    axes[0, 1].plot(pred_7step[:100], label='Predicted', linewidth=2, marker='s', markersize=3, alpha=0.7)
    axes[0, 1].set_title('7-Step Ahead Predictions (First 100 samples)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot 1-step
    axes[1, 0].scatter(actual_1step, pred_1step, alpha=0.5, s=20)
    axes[1, 0].plot([actual_1step.min(), actual_1step.max()], 
                    [actual_1step.min(), actual_1step.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_title('1-Step Ahead: Predicted vs Actual', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Actual Value')
    axes[1, 0].set_ylabel('Predicted Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot 7-step
    axes[1, 1].scatter(actual_7step, pred_7step, alpha=0.5, s=20, color='orange')
    axes[1, 1].plot([actual_7step.min(), actual_7step.max()], 
                    [actual_7step.min(), actual_7step.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_title('7-Step Ahead: Predicted vs Actual', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Actual Value')
    axes[1, 1].set_ylabel('Predicted Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_quality.png', dpi=300, bbox_inches='tight')
    print("✓ Prediction quality visualization saved")
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: Visualizing and Analyzing Attention Weights")
    print("=" * 60)
    
    # Load data
    print("\nLoading attention weights and predictions...")
    attention_weights, predictions, actuals = load_attention_data()
    print(f"✓ Loaded {len(attention_weights)} attention weight matrices")
    
    # Visualize attention heatmaps
    print("\nCreating attention heatmaps...")
    visualize_attention_heatmap(attention_weights, num_samples=6)
    
    # Analyze temporal patterns
    print("\nAnalyzing temporal attention patterns...")
    top_positions, top_weights = analyze_temporal_patterns(attention_weights)
    
    # Generate interpretation
    print("\nGenerating attention interpretation...")
    interpretation = create_attention_interpretation(top_positions, top_weights)
    
    # Save interpretation
    with open('attention_interpretation.txt', 'w', encoding='utf-8') as f:
        f.write(interpretation)
    print("✓ Interpretation saved to 'attention_interpretation.txt'")
    
    # Print interpretation
    print("\n" + interpretation)
    
    # Visualize prediction quality
    print("\nVisualizing prediction quality...")
    compare_predictions(predictions, actuals)
    
    print("\n" + "=" * 60)
    print("Attention analysis complete!")
    print("Next: Run '6_generate_report.py'")
    print("=" * 60)
