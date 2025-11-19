from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os, re
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("Loading TinyLlama model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.set_attn_implementation("eager")
print("Model loaded with attention tracking.\n")

prompt = """I will give you the following Python 3 code snippet.
Please predict what the printed output would be.

Code:
```python
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(result)
```"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=60)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("=== MODEL OUTPUT ===")
print(output_text)

match = re.findall(r"\b\d+(?:\.\d+)?\b", output_text)
pred_output = match[-1] if match else "N/A"
print("\nPredicted Output:", pred_output)

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attentions = outputs.attentions
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

save_dir = "./vis/enhanced"
os.makedirs(save_dir, exist_ok=True)

def normalize(attn):
    """Contrast stretching: highlight the top 5%"""
    flat = attn.flatten()
    threshold = np.percentile(flat, 90)
    attn = np.clip(attn, 0, threshold)
    attn = attn / threshold
    return attn

num_layers = 4
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i in range(num_layers):
    avg = attentions[i][0].mean(0).cpu().numpy()
    norm = normalize(avg)

    ax = axes[i]
    im = ax.imshow(norm, cmap="viridis", interpolation="nearest")

    ax.set_title(f"Layer {i}")
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=6)
    ax.set_yticklabels(tokens, fontsize=6)

    plt.colorbar(im, ax=ax)

plt.tight_layout()
save_path = os.path.join(save_dir, "enhanced_attention_layers0_3.jpg")
plt.savefig(save_path, dpi=200)
plt.close()

display(HTML(f"""
<h2>Enhanced Attention Visualization</h2>
<p><b>Model Output:</b> {pred_output}</p>
<p><b>Heatmap includes token labels + enhanced contrast</b></p>
<img src="{save_path}" width="900">
"""))

print("Saved:", save_path) 