"""
Generate attention heatmaps from Llama 3 extracted data
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_DIR = Path("web/data/llama3")
LAYER = 0  # Which layer to visualize (0-31)
HEAD = 0   # Which attention head to visualize (0-31)
SAMPLE_IDX = 0  # Which text sample to show (0-29)

def load_attention_data(layer, head):
    """Load attention data for a specific layer and head."""
    attention_file = DATA_DIR / "attention" / f"layer{layer}_head{head}.json"
    
    if not attention_file.exists():
        raise FileNotFoundError(f"Attention file not found: {attention_file}")
    
    with open(attention_file, 'r') as f:
        data = json.load(f)
    
    return data

def load_token_data():
    """Load token metadata."""
    token_file = DATA_DIR / "tokens.json"
    
    if not token_file.exists():
        raise FileNotFoundError(f"Token file not found: {token_file}")
    
    with open(token_file, 'r') as f:
        data = json.load(f)
    
    return data

def get_sample_attention(attention_data, token_data, sample_idx):
    """Extract attention matrix for a specific sample."""
    # Find tokens for this sample
    tokens_info = [t for t in token_data['tokens'] if t.get('sample_id') == sample_idx and t['type'] == 'query']
    
    if not tokens_info:
        raise ValueError(f"No tokens found for sample {sample_idx}")
    
    # Get token strings
    tokens = [t['token'] for t in tokens_info]
    num_tokens = len(tokens)
    
    # Get attention weights for this sample
    start_idx = tokens_info[0]['index']
    end_idx = start_idx + num_tokens
    
    attention_matrix = []
    for token_data_entry in attention_data['tokens'][start_idx:end_idx]:
        attn_weights = token_data_entry['attention'][:num_tokens]  # Only first num_tokens
        attention_matrix.append(attn_weights)
    
    return np.array(attention_matrix), tokens

def create_heatmap(attention_matrix, tokens, layer, head, sample_idx, save_path="attention_heatmap.png"):
    """Create and save attention heatmap."""
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(attention_matrix, 
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='viridis',
                cbar_kws={'label': 'Attention Weight'},
                square=True,
                linewidths=0.5,
                linecolor='white',
                vmin=0,
                vmax=attention_matrix.max())
    
    plt.title(f'Llama 3 Attention Heatmap\nLayer {layer}, Head {head}, Sample {sample_idx}', 
              fontsize=14, pad=20)
    plt.xlabel('Key Tokens', fontsize=12)
    plt.ylabel('Query Tokens', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {save_path}")
    
    # Also show it
    plt.show()
    
    return save_path

def create_multiple_heatmaps(num_layers=4, num_heads=4):
    """Create a grid of heatmaps for multiple layers and heads."""
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(20, 16))
    fig.suptitle(f'Llama 3 Attention Patterns - Sample {SAMPLE_IDX}', fontsize=16)
    
    # Load token data once
    token_data = load_token_data()
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            ax = axes[layer_idx, head_idx]
            
            try:
                # Load attention data
                attention_data = load_attention_data(layer_idx, head_idx)
                attention_matrix, tokens = get_sample_attention(attention_data, token_data, SAMPLE_IDX)
                
                # Plot heatmap
                im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
                ax.set_title(f'L{layer_idx} H{head_idx}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error\n{str(e)[:20]}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.tight_layout()
    save_path = f"attention_grid_sample{SAMPLE_IDX}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Grid heatmap saved to: {save_path}")
    plt.show()
    
    return save_path

def main():
    """Main function to generate heatmaps."""
    print("=" * 60)
    print("Llama 3 Attention Heatmap Generator")
    print("=" * 60)
    print()
    
    try:
        # Load data
        print(f"Loading attention data for Layer {LAYER}, Head {HEAD}...")
        attention_data = load_attention_data(LAYER, HEAD)
        token_data = load_token_data()
        
        # Extract attention matrix for sample
        print(f"Extracting attention matrix for sample {SAMPLE_IDX}...")
        attention_matrix, tokens = get_sample_attention(attention_data, token_data, SAMPLE_IDX)
        
        print(f"Attention matrix shape: {attention_matrix.shape}")
        print(f"Tokens: {tokens}")
        print()
        
        # Create single heatmap
        print("Creating heatmap...")
        save_path = create_heatmap(attention_matrix, tokens, LAYER, HEAD, SAMPLE_IDX)
        print()
        
        # Ask if user wants grid
        print("=" * 60)
        print("Creating grid of multiple attention heads...")
        create_multiple_heatmaps(num_layers=4, num_heads=4)
        print()
        
        print("=" * 60)
        print("✓ Done! Check the generated PNG files.")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
