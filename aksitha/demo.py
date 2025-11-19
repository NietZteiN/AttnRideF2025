from transformers import AutoTokenizer
from attention_renderer import AttentionRenderer

tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
renderer = AttentionRenderer(tok)

code = "def add(a, b):\n    return a + b"
instruction = "Summarize what this function does."

enc = tok(f"{instruction}\n\n```java\n{code}\n```", return_offsets_mapping=True)
n = len(enc["offset_mapping"])
fake_scores = [i/(n-1) for i in range(n)]

attn_map = renderer.map_attention_from_prompt_scores(
    code, instruction, fake_scores, pool_name="all_layers_mean"
)

renderer.render_html(code, attn_map, "out.html")
print("✅ Created out.html — open it to see the green attention shading!")

