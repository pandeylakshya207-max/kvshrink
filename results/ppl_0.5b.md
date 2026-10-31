# WikiText-2 perplexity: Qwen/Qwen2.5-0.5B-Instruct (float16)

GPU: Tesla T4 | torch 2.11.0+cu130 | transformers 5.22.1 | generated 2026-10-31T16:32:54+00:00

16 windows x 2048 tokens; prefix 128 tokens prefilled, remaining 1920 per window scored in chunks of 16 (30720 scored tokens). Deterministic: no sampling.

| method | spec | PPL | delta PPL vs full | KV bytes/token | compression vs full | peak allocated MiB | time s |
|---|---|---:|---:|---:|---:|---:|---:|
| full | type=full | 13.090 | +0.000 | 12288 | 1.00x | 1413 | 7 |
| int8 | type=quant, key_bits=8, value_bits=8 | 13.188 | +0.098 | 6528 | 1.88x | 1323 | 15 |
| kivi4 | type=kivi, bits=4, group_size=32, residual=128 | 13.143 | +0.053 | 4368 | 2.81x | 1294 | 14 |
| kivi2 | type=kivi, bits=2, group_size=32, residual=128 | 16.614 | +3.524 | 2928 | 4.20x | 1270 | 16 |
| stream1024 | type=streaming, budget=1024, sink=4 | 13.339 | +0.249 | 6144 | 2.00x | 1317 | 7 |
| stream512 | type=streaming, budget=512, sink=4 | 14.541 | +1.451 | 3072 | 4.00x | 1269 | 7 |
| h2o1024 | type=h2o, budget=1024, recent=512 | 13.174 | +0.085 | 6240 | 1.97x | 1317 | 9 |
| h2o512 | type=h2o, budget=512, recent=256 | 13.840 | +0.750 | 3120 | 3.94x | 1270 | 9 |

KV bytes/token = cache.used_bytes() / (batch x window), averaged over batches; includes scales, zero points and the fp16 residual window.
