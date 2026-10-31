# WikiText-2 perplexity: Qwen/Qwen2.5-0.5B-Instruct (float16)

GPU: Tesla T4 | torch 2.11.0+cu130 | transformers 5.22.1 | generated 2026-10-31T20:16:37+00:00

16 windows x 2048 tokens; prefix 128 tokens prefilled, remaining 1920 per window scored in chunks of 16 (30720 scored tokens). Deterministic: no sampling.

| method | spec | PPL | delta PPL vs full | KV bytes/token | compression vs full | peak allocated MiB | time s |
|---|---|---:|---:|---:|---:|---:|---:|
| full | type=full | 12.078 | +0.000 | 12288 | 1.00x | 1413 | 7 |
| kivi4 | type=kivi, bits=4, group_size=32, residual=128 | 12.109 | +0.031 | 4368 | 2.81x | 1294 | 15 |
| kivi2 | type=kivi, bits=2, group_size=32, residual=128 | 14.455 | +2.377 | 2928 | 4.20x | 1270 | 16 |
| k4v2 | type=quant, key_bits=4, value_bits=2, key_mode=channel, value_mode=token, key_group=32, value_group=32, residual=128 | 12.409 | +0.331 | 3648 | 3.37x | 1282 | 15 |
| mixed12 | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={0: {'key_bits': 2, 'value_bits': 2}, 1: {'key_bits': 2, 'value_bits': 2}, 2: {'key_bits': 2, 'value_bits': 2}, 5: {'key_bits': 2, 'value_bits': 2}, 6: {'key_bits': 2, 'value_bits': 2}, 12: {'key_bits': 2, 'value_bits': 2}, 13: {'key_bits': 2, 'value_bits': 2}, 14: {'key_bits': 2, 'value_bits': 2}, 17: {'key_bits': 2, 'value_bits': 2}, 18: {'key_bits': 2, 'value_bits': 2}, 19: {'key_bits': 2, 'value_bits': 2}, 23: {'key_bits': 2, 'value_bits': 2}} | 12.391 | +0.313 | 3648 | 3.37x | 1282 | 16 |

KV bytes/token = cache.used_bytes() / (batch x window), averaged over batches; includes scales, zero points and the fp16 residual window.
