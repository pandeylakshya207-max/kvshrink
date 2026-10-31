"""Print the KV-relevant config fields and fp16 KV bytes per token for each model."""
from transformers import AutoConfig

for name in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]:
    c = AutoConfig.from_pretrained(name)
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    kv_bytes = 2 * c.num_hidden_layers * c.num_key_value_heads * head_dim * 2  # K+V, fp16
    print(name)
    print(f"  layers={c.num_hidden_layers} q_heads={c.num_attention_heads} "
          f"kv_heads={c.num_key_value_heads} head_dim={head_dim} "
          f"max_pos={c.max_position_embeddings} rope={getattr(c, 'rope_parameters', None)} "
          f"sliding_window={getattr(c, 'use_sliding_window', None)}")
    print(f"  fp16 KV bytes/token = {kv_bytes}  ({kv_bytes/1024:.1f} KiB)")
    for ctx in (4096, 8192, 16384):
        print(f"  KV per sequence @ {ctx}: {kv_bytes*ctx/2**20:.0f} MiB")
