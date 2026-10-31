# Needle-in-a-haystack (passkey retrieval): Qwen/Qwen2.5-0.5B-Instruct (float16)

GPU: Tesla T4 | torch 2.11.0+cu130 | transformers 5.22.1 | generated 2026-10-31T19:45:33+00:00

Depths [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]; seeds [0, 1, 2]; chunked prefill of 512 tokens for every method; greedy decoding of 10 tokens. Cells: accuracy, mean over seeds +/- 95% t-CI (per-seed accuracy = mean over depths). In brackets: KV compression vs full.

| method | spec | 8192 |
|---|---|---:|
| full | type=full | 100% +/- 0 (1.00x) |
| kivi2 | type=kivi, bits=2, group_size=32, residual=128 | 63% +/- 29 (4.98x) |
| kivi2_r0 | type=kivi, bits=2, group_size=32, residual=0 | 60% +/- 50 (5.32x) |
| kivi2_r32 | type=kivi, bits=2, group_size=32, residual=32 | 63% +/- 14 (5.23x) |
| kivi2_r512 | type=kivi, bits=2, group_size=32, residual=512 | 67% +/- 38 (4.19x) |
| kivi2_g16 | type=kivi, bits=2, group_size=16, residual=128 | 93% +/- 29 (3.82x) |
| kivi2_g64 | type=kivi, bits=2, group_size=64, residual=128 | 23% +/- 80 (5.89x) |
| k4v2 | type=quant, key_bits=4, value_bits=2, key_mode=channel, value_mode=token, key_group=32, value_group=32, residual=128 | 100% +/- 0 (3.82x) |
| k2v4 | type=quant, key_bits=2, value_bits=4, key_mode=channel, value_mode=token, key_group=32, value_group=32, residual=128 | 73% +/- 38 (3.81x) |
| q2_tokenkeys | type=quant, key_bits=2, value_bits=2, key_mode=token, value_mode=token, key_group=32, value_group=32, residual=128 | 0% +/- 0 (5.00x) |
| kivi4 | type=kivi, bits=4, group_size=32, residual=128 | 100% +/- 0 (3.09x) |
| q4_tokenkeys | type=quant, key_bits=4, value_bits=4, key_mode=token, value_mode=token, key_group=32, value_group=32, residual=128 | 0% +/- 0 (3.09x) |

## Accuracy by needle depth (pooled over seeds)

| method | context | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 8192 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| kivi2 | 8192 | 100% | 67% | 100% | 67% | 67% | 33% | 0% | 33% | 67% | 100% |
| kivi2_r0 | 8192 | 100% | 67% | 100% | 33% | 67% | 33% | 33% | 67% | 67% | 33% |
| kivi2_r32 | 8192 | 100% | 33% | 67% | 67% | 100% | 100% | 33% | 67% | 33% | 33% |
| kivi2_r512 | 8192 | 100% | 100% | 0% | 33% | 100% | 67% | 67% | 100% | 33% | 67% |
| kivi2_g16 | 8192 | 100% | 100% | 100% | 100% | 100% | 100% | 67% | 67% | 100% | 100% |
| kivi2_g64 | 8192 | 67% | 33% | 0% | 0% | 33% | 0% | 0% | 33% | 33% | 33% |
| k4v2 | 8192 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| k2v4 | 8192 | 100% | 67% | 67% | 100% | 67% | 33% | 100% | 100% | 33% | 67% |
| q2_tokenkeys | 8192 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| kivi4 | 8192 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| q4_tokenkeys | 8192 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
