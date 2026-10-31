# WikiText-2 perplexity: Qwen/Qwen2.5-0.5B-Instruct (float16)

GPU: Tesla T4 | torch 2.11.0+cu130 | transformers 5.22.1 | generated 2026-10-31T19:56:28+00:00

16 windows x 2048 tokens; prefix 128 tokens prefilled, remaining 1920 per window scored in chunks of 16 (30720 scored tokens). Deterministic: no sampling.

| method | spec | PPL | delta PPL vs full | KV bytes/token | compression vs full | peak allocated MiB | time s |
|---|---|---:|---:|---:|---:|---:|---:|
| full | type=full | 13.090 | +0.000 | 12288 | 1.00x | 1413 | 7 |
| kivi4 | type=kivi, bits=4, group_size=32, residual=128 | 13.143 | +0.053 | 4368 | 2.81x | 1294 | 15 |
| kivi2 | type=kivi, bits=2, group_size=32, residual=128 | 16.614 | +3.524 | 2928 | 4.20x | 1270 | 16 |
| l00_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={0: {'key_bits': 2, 'value_bits': 2}} | 13.151 | +0.061 | 4308 | 2.85x | 1293 | 15 |
| l01_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={1: {'key_bits': 2, 'value_bits': 2}} | 13.139 | +0.050 | 4308 | 2.85x | 1293 | 15 |
| l02_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={2: {'key_bits': 2, 'value_bits': 2}} | 13.158 | +0.068 | 4308 | 2.85x | 1293 | 15 |
| l03_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={3: {'key_bits': 2, 'value_bits': 2}} | 13.234 | +0.144 | 4308 | 2.85x | 1293 | 15 |
| l04_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={4: {'key_bits': 2, 'value_bits': 2}} | 13.207 | +0.118 | 4308 | 2.85x | 1293 | 15 |
| l05_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={5: {'key_bits': 2, 'value_bits': 2}} | 13.173 | +0.083 | 4308 | 2.85x | 1293 | 15 |
| l06_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={6: {'key_bits': 2, 'value_bits': 2}} | 13.201 | +0.111 | 4308 | 2.85x | 1293 | 15 |
| l07_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={7: {'key_bits': 2, 'value_bits': 2}} | 13.204 | +0.114 | 4308 | 2.85x | 1293 | 15 |
| l08_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={8: {'key_bits': 2, 'value_bits': 2}} | 13.223 | +0.133 | 4308 | 2.85x | 1293 | 15 |
| l09_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={9: {'key_bits': 2, 'value_bits': 2}} | 13.444 | +0.354 | 4308 | 2.85x | 1293 | 15 |
| l10_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={10: {'key_bits': 2, 'value_bits': 2}} | 13.286 | +0.196 | 4308 | 2.85x | 1293 | 15 |
| l11_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={11: {'key_bits': 2, 'value_bits': 2}} | 13.339 | +0.249 | 4308 | 2.85x | 1293 | 15 |
| l12_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={12: {'key_bits': 2, 'value_bits': 2}} | 13.169 | +0.080 | 4308 | 2.85x | 1293 | 15 |
| l13_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={13: {'key_bits': 2, 'value_bits': 2}} | 13.203 | +0.113 | 4308 | 2.85x | 1293 | 15 |
| l14_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={14: {'key_bits': 2, 'value_bits': 2}} | 13.190 | +0.101 | 4308 | 2.85x | 1293 | 15 |
| l15_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={15: {'key_bits': 2, 'value_bits': 2}} | 13.216 | +0.126 | 4308 | 2.85x | 1293 | 15 |
| l16_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={16: {'key_bits': 2, 'value_bits': 2}} | 13.429 | +0.339 | 4308 | 2.85x | 1293 | 15 |
| l17_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={17: {'key_bits': 2, 'value_bits': 2}} | 13.158 | +0.068 | 4308 | 2.85x | 1293 | 15 |
| l18_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={18: {'key_bits': 2, 'value_bits': 2}} | 13.162 | +0.073 | 4308 | 2.85x | 1293 | 15 |
| l19_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={19: {'key_bits': 2, 'value_bits': 2}} | 13.173 | +0.083 | 4308 | 2.85x | 1293 | 15 |
| l20_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={20: {'key_bits': 2, 'value_bits': 2}} | 13.238 | +0.149 | 4308 | 2.85x | 1293 | 15 |
| l21_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={21: {'key_bits': 2, 'value_bits': 2}} | 13.249 | +0.159 | 4308 | 2.85x | 1293 | 15 |
| l22_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={22: {'key_bits': 2, 'value_bits': 2}} | 13.231 | +0.141 | 4308 | 2.85x | 1293 | 15 |
| l23_2bit | type=kivi, bits=4, group_size=32, residual=128, layer_overrides={23: {'key_bits': 2, 'value_bits': 2}} | 13.165 | +0.075 | 4308 | 2.85x | 1293 | 15 |

KV bytes/token = cache.used_bytes() / (batch x window), averaged over batches; includes scales, zero points and the fp16 residual window.
