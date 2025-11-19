"""
Extract attention patterns from Llama 3 model for visualization in attention-viz.
This script generates the required JSON files in the format expected by the visualization tool.
"""

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.manifold import TSNE
import umap
from pathlib import Path
from tqdm import tqdm
import os
import gc

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # Change to "meta-llama/Meta-Llama-3-70B" for larger model
OUTPUT_DIR = Path("web/data/llama3")
NUM_SAMPLES = 30  # Number of text samples to process (reduced for 7GB VRAM)
MAX_LENGTH = 64  # Max sequence length (reduced for 7GB VRAM)
USE_UMAP = True  # Set to False to use t-SNE instead
USE_4BIT_QUANTIZATION = True  # Use 4-bit quantization to reduce memory usage

# Sample texts for analysis (you can replace with your own dataset)
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming the world of artificial intelligence.",
    "Python is a versatile programming language used in data science.",
    "The capital of France is Paris, a beautiful city known for art and culture.",
    "Climate change is one of the most pressing issues of our time.",
    "Neural networks are inspired by the structure of the human brain.",
    "The internet has revolutionized how we communicate and share information.",
    "Renewable energy sources like solar and wind are becoming more affordable.",
    "Space exploration continues to push the boundaries of human knowledge.",
    "Quantum computing promises to solve problems that are intractable for classical computers.",
]

def load_model():
    """Load Llama 3 model and tokenizer."""
    print(f"Loading {MODEL_NAME}...")
    print("Using 4-bit quantization to fit in 7GB VRAM...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if USE_4BIT_QUANTIZATION:
        # Configure 4-bit quantization for low memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            output_attentions=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            output_attentions=True,
        )
    
    model.eval()
    print("Model loaded successfully!")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return model, tokenizer

def extract_attention_data(model, tokenizer, texts):
    """Extract attention weights and hidden states from the model."""
    all_attentions = []
    all_tokens = []
    all_hidden_states = []
    
    print("Extracting attention data...")
    for idx, text in enumerate(tqdm(texts[:NUM_SAMPLES])):
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Store data
        all_tokens.append({
            'text': text,
            'tokens': tokens,
            'token_ids': inputs['input_ids'][0].cpu().tolist()
        })
        
        # Get attention weights (list of tensors, one per layer)
        attentions = [attn[0].cpu() for attn in outputs.attentions]  # [0] for batch
        all_attentions.append(attentions)
        
        # Get hidden states for query/key extraction
        hidden_states = [h[0].cpu() for h in outputs.hidden_states]
        all_hidden_states.append(hidden_states)
        
        # Clear GPU cache every 5 samples to prevent OOM
        if (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return all_attentions, all_tokens, all_hidden_states

def compute_embeddings(queries, keys, method='umap'):
    """Compute 2D embeddings of query-key pairs."""
    # Flatten all queries and keys
    all_queries = torch.cat([q.reshape(-1, q.shape[-1]) for q in queries], dim=0).numpy()
    all_keys = torch.cat([k.reshape(-1, k.shape[-1]) for k in keys], dim=0).numpy()
    
    # Concatenate query-key pairs
    qk_pairs = np.concatenate([all_queries, all_keys], axis=1)
    
    print(f"Computing {method.upper()} embeddings...")
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42)
    
    embeddings = reducer.fit_transform(qk_pairs)
    return embeddings

def generate_attention_json_files(model, all_attentions, all_tokens):
    """Generate attention JSON files for each layer and head."""
    num_layers = len(all_attentions[0])
    num_heads = all_attentions[0][0].shape[0]  # First dimension is num_heads
    
    attention_dir = OUTPUT_DIR / "attention"
    attention_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating attention JSON files...")
    for layer_idx in tqdm(range(num_layers)):
        for head_idx in range(num_heads):
            attention_data = {
                'layer': layer_idx,
                'head': head_idx,
                'tokens': []
            }
            
            # Collect attention weights for this layer/head across all samples
            token_idx = 0
            for sample_idx, attentions in enumerate(all_attentions):
                layer_attn = attentions[layer_idx]  # [num_heads, seq_len, seq_len]
                head_attn = layer_attn[head_idx]  # [seq_len, seq_len]
                
                tokens = all_tokens[sample_idx]['tokens']
                seq_len = len(tokens)
                
                # For each token (query), store its attention weights
                for i in range(seq_len):
                    attention_weights = head_attn[i].tolist()
                    attention_data['tokens'].append({
                        'index': token_idx,
                        'attention': attention_weights,
                        'token': tokens[i]
                    })
                    token_idx += 1
            
            # Save to JSON
            filename = f"layer{layer_idx}_head{head_idx}.json"
            with open(attention_dir / filename, 'w') as f:
                json.dump(attention_data, f)

def generate_matrix_data(model, all_attentions, all_tokens, all_hidden_states):
    """Generate matrix data with embeddings for each layer/head."""
    num_layers = len(all_attentions[0])
    num_heads = all_attentions[0][0].shape[0]
    
    matrix_dir = OUTPUT_DIR / "byLayerHead"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating matrix data with embeddings...")
    print(f"Processing {num_layers} layers × {num_heads} heads = {num_layers * num_heads} files...")
    
    # Get model dimensions
    hidden_dim = all_hidden_states[0][0].shape[-1]
    head_dim = hidden_dim // num_heads
    
    # Process in smaller batches to save memory
    for layer_idx in tqdm(range(num_layers)):
        for head_idx in range(num_heads):
            # Extract query and key vectors for this head
            queries = []
            keys = []
            
            for sample_idx, hidden_states in enumerate(all_hidden_states):
                # Get hidden state for this layer
                layer_hidden = hidden_states[layer_idx + 1]  # +1 because hidden_states includes input embeddings
                
                # Split into heads and extract Q, K
                # This is a simplification - actual Q, K matrices would need the model's projection weights
                # For visualization purposes, we use the hidden states directly
                seq_len = layer_hidden.shape[0]
                reshaped = layer_hidden.reshape(seq_len, num_heads, head_dim)
                
                # Use the hidden state as approximation for Q and K
                queries.append(reshaped[:, head_idx, :])
                keys.append(reshaped[:, head_idx, :])
            
            # Compute embeddings
            method = 'umap' if USE_UMAP else 'tsne'
            embeddings = compute_embeddings(queries, keys, method=method)
            
            # Compute norms
            all_q = torch.cat(queries, dim=0)
            q_norms = torch.norm(all_q, dim=1).tolist()
            
            # Create matrix data
            matrix_data = {
                'layer': layer_idx,
                'head': head_idx,
                'embeddings': embeddings.tolist(),
                'q_norms': q_norms,
                'method': method
            }
            
            # Save to JSON
            filename = f"layer{layer_idx}_head{head_idx}.json"
            with open(matrix_dir / filename, 'w') as f:
                json.dump(matrix_data, f)
            
            # Clear memory periodically
            if (head_idx + 1) % 8 == 0:
                gc.collect()

def generate_token_data(all_tokens):
    """Generate tokens.json with metadata for all tokens."""
    print("Generating token data...")
    
    all_tokens_flat = []
    token_idx = 0
    
    for sample_idx, sample_data in enumerate(all_tokens):
        tokens = sample_data['tokens']
        text = sample_data['text']
        
        for i, token in enumerate(tokens):
            all_tokens_flat.append({
                'index': token_idx,
                'token': token,
                'sentence': text,
                'pos_int': i,
                'length': len(tokens),
                'type': 'query',  # Mark as query
                'sample_id': sample_idx
            })
            token_idx += 1
        
        # Add corresponding key tokens
        for i, token in enumerate(tokens):
            all_tokens_flat.append({
                'index': token_idx,
                'token': token,
                'sentence': text,
                'pos_int': i,
                'length': len(tokens),
                'type': 'key',  # Mark as key
                'sample_id': sample_idx
            })
            token_idx += 1
    
    token_data = {'tokens': all_tokens_flat}
    
    with open(OUTPUT_DIR / "tokens.json", 'w') as f:
        json.dump(token_data, f)

def generate_agg_attention(all_attentions, all_tokens):
    """Generate aggregate attention data."""
    print("Generating aggregate attention data...")
    
    num_layers = len(all_attentions[0])
    num_heads = all_attentions[0][0].shape[0]
    
    agg_data = {}
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            key = f"{layer_idx}_{head_idx}"
            
            # Aggregate attention across all samples
            all_attn_for_head = []
            for attentions in all_attentions:
                layer_attn = attentions[layer_idx]
                head_attn = layer_attn[head_idx]
                
                # Average attention across sequence
                avg_attn = head_attn.mean(dim=0).tolist()
                all_attn_for_head.append(avg_attn)
            
            agg_data[key] = all_attn_for_head
    
    with open(OUTPUT_DIR / "agg_attn.json", 'w') as f:
        json.dump(agg_data, f)

def main():
    """Main extraction pipeline."""
    print("=" * 50)
    print("Llama 3 Attention Data Extraction")
    print("=" * 50)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model()
    
    # Prepare texts
    texts = SAMPLE_TEXTS * (NUM_SAMPLES // len(SAMPLE_TEXTS) + 1)
    texts = texts[:NUM_SAMPLES]
    
    # Extract attention data
    all_attentions, all_tokens, all_hidden_states = extract_attention_data(model, tokenizer, texts)
    
    # Generate JSON files
    generate_attention_json_files(model, all_attentions, all_tokens)
    generate_matrix_data(model, all_attentions, all_tokens, all_hidden_states)
    generate_token_data(all_tokens)
    generate_agg_attention(all_attentions, all_tokens)
    
    print("\n" + "=" * 50)
    print(f"✓ Data extraction complete!")
    print(f"✓ Files saved to: {OUTPUT_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()
