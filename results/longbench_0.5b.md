# LongBench subset: Qwen/Qwen2.5-0.5B-Instruct (float16)

GPU: Tesla T4 | torch 2.11.0+cu130 | transformers 5.22.1 | generated 2026-10-31T19:05:38+00:00

30 examples per task (seeded sample, same for all methods); prompts middle-truncated to 7500 tokens; chunked prefill of 512; greedy decoding. Scores 0-100, cells: mean [95% bootstrap CI].

| method | qasper | hotpotqa | triviaqa | passage_retrieval_en | average | compression vs full |
|---|---:|---:|---:|---:|---:|---:|
| full | 24.9 [14.7, 36.4] | 21.9 [9.4, 35.9] | 64.4 [49.4, 78.6] | 6.7 [0.0, 16.7] | 29.5 | 1.00x |
| int8 | 24.4 [13.4, 37.1] | 20.6 [9.4, 34.1] | 61.8 [47.3, 76.4] | 3.3 [0.0, 10.0] | 27.5 | 1.88x |
| kivi4 | 30.6 [19.1, 43.4] | 20.6 [8.5, 34.2] | 64.2 [49.0, 78.9] | 3.3 [0.0, 10.0] | 29.7 | 3.05x |
| kivi2 | 21.6 [10.1, 35.2] | 16.4 [5.7, 29.2] | 41.6 [26.1, 58.0] | 3.3 [0.0, 10.0] | 20.7 | 4.87x |
| stream50 | 21.8 [10.9, 35.0] | 24.4 [10.3, 40.6] | 48.5 [35.1, 62.2] | 0.0 [0.0, 0.0] | 23.7 | 2.00x |
| stream25 | 17.8 [7.0, 30.8] | 23.5 [10.2, 37.7] | 47.8 [33.4, 62.0] | 0.0 [0.0, 0.0] | 22.3 | 4.01x |
| h2o50 | 23.9 [11.9, 38.4] | 21.3 [9.0, 35.0] | 55.4 [40.5, 70.7] | 3.3 [0.0, 10.0] | 26.0 | 1.97x |
| h2o25 | 18.8 [8.0, 32.0] | 21.7 [8.9, 35.5] | 56.1 [41.0, 72.1] | 3.3 [0.0, 10.0] | 25.0 | 3.94x |
