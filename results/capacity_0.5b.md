# Decode capacity per KV-cache method: Qwen/Qwen2.5-0.5B-Instruct

GPU: Tesla T4 (14.6 GiB) | weights 0.93 GiB | torch 2.11.0+cu130 | generated 2026-10-31T20:30:18+00:00

Measured: largest batch that decodes without OOM, cache filled synthetically to the context length (contents are zeros; memory and compute do not depend on them). Ideal (ANALYTIC, not measured): (total - weights - 0.5 GiB reserve) / measured KV bytes per sequence.

| method | context | measured max batch | vs full | ideal max batch (analytic) | vs full | decode tok/s at max batch | peak allocated GiB | KV GiB at max batch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 4096 | 284 | 1.00x | 280 | 1.00x | 3453 | 14.33 | 13.31 |
| full | 8192 | 142 | 1.00x | 140 | 1.00x | 1767 | 14.29 | 13.31 |
| int8 | 4096 | 488 | 1.72x | 527 | 1.88x | 220 | 14.14 | 12.15 |
| int8 | 8192 | 242 | 1.70x | 263 | 1.88x | 111 | 14.12 | 12.05 |
| kivi4 | 4096 | 712 | 2.51x | 839 | 3.00x | 197 | 14.03 | 11.15 |
| kivi4 | 8192 | 376 | 2.65x | 433 | 3.09x | 100 | 14.28 | 11.39 |
| k4v2 | 4096 | 848 | 2.99x | 1024 | 3.66x | 193 | 14.02 | 10.87 |
| k4v2 | 8192 | 448 | 3.15x | 535 | 3.82x | 96 | 14.13 | 10.99 |
| kivi2 | 4096 | 1056 | 3.72x | 1316 | 4.70x | 187 | 14.23 | 10.54 |
| kivi2 | 8192 | 561 | 3.95x | 699 | 4.99x | 92 | 14.20 | 10.53 |
| stream50 | 4096 | 528 | 1.86x | 560 | 2.00x | 1214 | 14.35 | 12.38 |
| stream50 | 8192 | 264 | 1.86x | 280 | 2.00x | 626 | 14.34 | 12.38 |
| stream25 | 4096 | 1048 | 3.69x | 1120 | 4.00x | 2240 | 14.25 | 12.28 |
| stream25 | 8192 | 528 | 3.72x | 560 | 4.00x | 1189 | 14.35 | 12.38 |
| h2o50 | 4096 | 511 | 1.80x | 551 | 1.97x | 666 | 13.68 | 12.16 |
| h2o50 | 8192 | 255 | 1.80x | 275 | 1.96x | 337 | 13.68 | 12.14 |
| h2o25 | 4096 | 1019 | 3.59x | 1103 | 3.94x | 1230 | 13.64 | 12.13 |
| h2o25 | 8192 | 511 | 3.60x | 551 | 3.94x | 664 | 13.68 | 12.16 |
