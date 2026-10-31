# Baseline: Qwen/Qwen2.5-0.5B-Instruct (float16, full-precision KV cache)

GPU: Tesla T4 | torch 2.11.0+cu130 (CUDA 13.0) | transformers 5.22.1 | generated 2026-10-31T14:18:04+00:00

## KV cache size

- Measured (torch.cuda.memory_allocated delta over 8192 tokens): 12288 bytes/token
- Analytic (2 x layers x kv_heads x head_dim x bytes): 12288 bytes/token
- Model weights allocated: 950 MiB

## Batch-1 latency (random token ids; mean +/- std over 3 runs)

| context | prefill tok/s | decode tok/s | decode ms/token | peak allocated MiB | reserved MiB | nvidia-smi MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 17748.1 +/- 1641.0 | 43.2 +/- 0.7 | 23.2 +/- 0.4 | 983 | 1048 | 1181 |
| 2048 | 19980.7 +/- 310.3 | 43.3 +/- 0.7 | 23.1 +/- 0.4 | 1052 | 1128 | 1261 |
| 4096 | 17660.1 +/- 165.0 | 42.9 +/- 0.5 | 23.3 +/- 0.3 | 1142 | 1202 | 1335 |
| 8192 | 14142.8 +/- 264.2 | 43.5 +/- 0.7 | 23.0 +/- 0.4 | 1315 | 1450 | 1583 |

## Decode capacity (SYNTHETIC cache fill; prefill memory not included)

| context | max batch | decode tok/s at max batch | peak allocated MiB | KV cache MiB |
|---:|---:|---:|---:|---:|
| 4096 | 284 | 3107 | 14675 | 13632 |
| 8192 | 142 | 1598 | 14634 | 13632 |
