# WikiText-2 perplexity: Qwen/Qwen2.5-0.5B-Instruct (float16)

GPU: Tesla T4 | torch 2.11.0+cu130 | transformers 5.22.1 | generated 2026-10-31T19:42:02+00:00

16 windows x 2048 tokens; prefix 128 tokens prefilled, remaining 1920 per window scored in chunks of 16 (30720 scored tokens). Deterministic: no sampling.

| method | spec | PPL | delta PPL vs full | KV bytes/token | compression vs full | peak allocated MiB | time s |
|---|---|---:|---:|---:|---:|---:|---:|
| full | type=full | 13.090 | +0.000 | 12288 | 1.00x | 1413 | 7 |
| kivi2 | type=kivi, bits=2, group_size=32, residual=128 | 16.614 | +3.524 | 2928 | 4.20x | 1270 | 16 |
| kivi2_r0 | type=kivi, bits=2, group_size=32, residual=0 | 17.920 | +4.830 | 2304 | 5.33x | 1258 | 17 |
| kivi2_r32 | type=kivi, bits=2, group_size=32, residual=32 | 17.044 | +3.955 | 2460 | 5.00x | 1261 | 17 |
| kivi2_r512 | type=kivi, bits=2, group_size=32, residual=512 | 15.477 | +2.387 | 4800 | 2.56x | 1309 | 15 |
| kivi2_g16 | type=kivi, bits=2, group_size=16, residual=128 | 14.793 | +1.703 | 3648 | 3.37x | 1281 | 18 |
| kivi2_g64 | type=kivi, bits=2, group_size=64, residual=128 | 19.665 | +6.576 | 2568 | 4.79x | 1265 | 16 |
| k4v2 | type=quant, key_bits=4, value_bits=2, key_mode=channel, value_mode=token, key_group=32, value_group=32, residual=128 | 13.467 | +0.377 | 3648 | 3.37x | 1282 | 16 |
| k2v4 | type=quant, key_bits=2, value_bits=4, key_mode=channel, value_mode=token, key_group=32, value_group=32, residual=128 | 15.967 | +2.878 | 3648 | 3.37x | 1282 | 15 |
| q2_tokenkeys | type=quant, key_bits=2, value_bits=2, key_mode=token, value_mode=token, key_group=32, value_group=32, residual=128 | 326.207 | +313.117 | 2928 | 4.20x | 1269 | 18 |
| kivi4 | type=kivi, bits=4, group_size=32, residual=128 | 13.143 | +0.053 | 4368 | 2.81x | 1294 | 15 |
| q4_tokenkeys | type=quant, key_bits=4, value_bits=4, key_mode=token, value_mode=token, key_group=32, value_group=32, residual=128 | 25.380 | +12.291 | 4368 | 2.81x | 1293 | 16 |

KV bytes/token = cache.used_bytes() / (batch x window), averaged over batches; includes scales, zero points and the fp16 residual window.
