# Needle-in-a-haystack (passkey retrieval): Qwen/Qwen2.5-0.5B-Instruct (float16)

GPU: Tesla T4 | torch 2.11.0+cu130 | transformers 5.22.1 | generated 2026-10-31T16:58:26+00:00

Depths [0.1, 0.3, 0.5, 0.7, 0.9]; seeds [0, 1, 2]; chunked prefill of 512 tokens for every method; greedy decoding of 10 tokens. Cells: accuracy, mean over seeds +/- 95% t-CI (per-seed accuracy = mean over depths). In brackets: KV compression vs full.

| method | spec | 4096 | 8192 | 16384 |
|---|---|---:|---:|---:|
| full | type=full | 100% +/- 0 (1.00x) | 100% +/- 0 (1.00x) | 100% +/- 0 (1.00x) |
| int8 | type=quant, key_bits=8, value_bits=8 | 100% +/- 0 (1.88x) | 100% +/- 0 (1.88x) | 100% +/- 0 (1.88x) |
| kivi4 | type=kivi, bits=4, group_size=32, residual=128 | 100% +/- 0 (2.99x) | 100% +/- 0 (3.09x) | 100% +/- 0 (3.14x) |
| kivi2 | type=kivi, bits=2, group_size=32, residual=128 | 60% +/- 50 (4.68x) | 53% +/- 29 (4.98x) | 47% +/- 76 (5.15x) |
| stream50 | type=streaming, budget_frac=0.5, sink=4 | 60% +/- 0 (2.00x) | 60% +/- 0 (2.00x) | 60% +/- 0 (2.00x) |
| stream25 | type=streaming, budget_frac=0.25, sink=4 | 20% +/- 0 (4.01x) | 20% +/- 0 (4.00x) | 20% +/- 0 (4.00x) |
| h2o50 | type=h2o, budget_frac=0.5, recent_frac=0.5 | 93% +/- 29 (1.97x) | 93% +/- 29 (1.97x) | 93% +/- 29 (1.97x) |
| h2o25 | type=h2o, budget_frac=0.25, recent_frac=0.5 | 20% +/- 0 (3.95x) | 27% +/- 29 (3.94x) | 20% +/- 0 (3.94x) |

## Accuracy by needle depth (pooled over seeds)

| method | context | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| full | 4096 | 100% | 100% | 100% | 100% | 100% |
| full | 8192 | 100% | 100% | 100% | 100% | 100% |
| full | 16384 | 100% | 100% | 100% | 100% | 100% |
| int8 | 4096 | 100% | 100% | 100% | 100% | 100% |
| int8 | 8192 | 100% | 100% | 100% | 100% | 100% |
| int8 | 16384 | 100% | 100% | 100% | 100% | 100% |
| kivi4 | 4096 | 100% | 100% | 100% | 100% | 100% |
| kivi4 | 8192 | 100% | 100% | 100% | 100% | 100% |
| kivi4 | 16384 | 100% | 100% | 100% | 100% | 100% |
| kivi2 | 4096 | 67% | 67% | 67% | 67% | 33% |
| kivi2 | 8192 | 100% | 33% | 67% | 33% | 33% |
| kivi2 | 16384 | 33% | 0% | 100% | 33% | 67% |
| stream50 | 4096 | 0% | 0% | 100% | 100% | 100% |
| stream50 | 8192 | 0% | 0% | 100% | 100% | 100% |
| stream50 | 16384 | 0% | 0% | 100% | 100% | 100% |
| stream25 | 4096 | 0% | 0% | 0% | 0% | 100% |
| stream25 | 8192 | 0% | 0% | 0% | 0% | 100% |
| stream25 | 16384 | 0% | 0% | 0% | 0% | 100% |
| h2o50 | 4096 | 67% | 100% | 100% | 100% | 100% |
| h2o50 | 8192 | 100% | 100% | 100% | 67% | 100% |
| h2o50 | 16384 | 67% | 100% | 100% | 100% | 100% |
| h2o25 | 4096 | 0% | 0% | 0% | 0% | 100% |
| h2o25 | 8192 | 0% | 0% | 33% | 0% | 100% |
| h2o25 | 16384 | 0% | 0% | 0% | 0% | 100% |
