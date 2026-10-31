# Needle-in-a-haystack (passkey retrieval): Qwen/Qwen2.5-0.5B-Instruct (float16)

GPU: Tesla T4 | torch 2.11.0+cu130 | transformers 5.22.1 | generated 2026-10-31T20:18:10+00:00

Depths [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]; seeds [0, 1, 2]; chunked prefill of 512 tokens for every method; greedy decoding of 10 tokens. Cells: accuracy, mean over seeds +/- 95% t-CI (per-seed accuracy = mean over depths). In brackets: KV compression vs full.

| method | spec | 8192 |
|---|---|---:|
| full | type=full | 100% +/- 0 (1.00x) |
| kivi4 | type=kivi, bits=4, group_size=32, residual=128 | 100% +/- 0 (3.09x) |
| kivi2 | type=kivi, bits=2, group_size=32, residual=128 | 63% +/- 29 (4.98x) |
| k4v2 | type=quant, key_bits=4, value_bits=2, key_mode=channel, value_mode=token, key_group=32, value_group=32, residual=128 | 100% +/- 0 (3.82x) |
| mixed12 | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={0: {'key_bits': 2, 'value_bits': 2}, 1: {'key_bits': 2, 'value_bits': 2}, 2: {'key_bits': 2, 'value_bits': 2}, 5: {'key_bits': 2, 'value_bits': 2}, 6: {'key_bits': 2, 'value_bits': 2}, 12: {'key_bits': 2, 'value_bits': 2}, 13: {'key_bits': 2, 'value_bits': 2}, 14: {'key_bits': 2, 'value_bits': 2}, 17: {'key_bits': 2, 'value_bits': 2}, 18: {'key_bits': 2, 'value_bits': 2}, 19: {'key_bits': 2, 'value_bits': 2}, 23: {'key_bits': 2, 'value_bits': 2}} | 80% +/- 66 (3.82x) |

## Accuracy by needle depth (pooled over seeds)

| method | context | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 8192 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| kivi4 | 8192 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| kivi2 | 8192 | 100% | 67% | 100% | 67% | 67% | 33% | 0% | 33% | 67% | 100% |
| k4v2 | 8192 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| mixed12 | 8192 | 67% | 100% | 100% | 100% | 67% | 100% | 67% | 67% | 33% | 100% |
