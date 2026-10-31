# WikiText-2 perplexity: Qwen/Qwen2.5-0.5B-Instruct (float16)

GPU: Tesla T4 | torch 2.11.0+cu130 | transformers 5.22.1 | generated 2026-10-31T16:14:40+00:00

16 windows x 2048 tokens; prefix 128 tokens prefilled, remaining 1920 per window scored in chunks of 16 (30720 scored tokens). Deterministic: no sampling.

| method | spec | PPL | delta PPL vs full | KV bytes/token | compression vs full | peak allocated MiB | time s |
|---|---|---:|---:|---:|---:|---:|---:|
| full | type=full | 13.090 | +0.000 | 12288 | 1.00x | 1413 | 8 |
| int8 | type=quant, key_bits=8, value_bits=8 | 13.188 | +0.098 | 6528 | 1.88x | 1323 | 15 |
| kivi4 | type=kivi, bits=4, group_size=32, residual=128 | 13.143 | +0.053 | 4368 | 2.81x | 1294 | 15 |
| kivi2 | type=kivi, bits=2, group_size=32, residual=128 | 16.614 | +3.524 | 2928 | 4.20x | 1270 | 16 |

KV bytes/token = cache.used_bytes() / (batch x window), averaged over batches; includes scales, zero points and the fp16 residual window.
