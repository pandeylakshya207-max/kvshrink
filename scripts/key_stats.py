"""Per-layer key-cache statistics: tests why per-token key quantization fails here.

Runs one WikiText-2 window through the model with a full-precision cache, then for
every layer, on the cached (post-RoPE) keys, measures:
  * outlier ratio: largest / median mean magnitude over RoPE channel pairs (d, d + D/2);
  * bias share: |k_proj bias| of that largest pair / its mean magnitude;
  * relative quantization error ||K - Q(K)|| / ||K|| for per-token and per-channel
    grouping (group 32, as in the ablations) at 2 and 4 bits.

Usage: python scripts/key_stats.py --output results/key_stats_0.5b
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvshrink.model import QwenRunner
from kvshrink.quant import dequantize, quantize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--group", type=int, default=32)
    p.add_argument("--output", default="results/key_stats_0.5b")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    runner = QwenRunner(model)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[:, : args.tokens].to(device)
    cache = runner.new_cache(1, args.tokens)
    runner.forward(ids, cache, last_only=True)

    rows = []
    for layer, block in enumerate(model.model.layers):
        K = cache.k[layer][0, :, : args.tokens].float()  # [H_kv, T, D]
        H, _, D = K.shape
        half = D // 2
        pair = torch.sqrt(K[..., :half] ** 2 + K[..., half:] ** 2).mean(dim=1)  # [H_kv, D/2]
        h, d = divmod(int(pair.argmax()), half)
        bias = block.self_attn.k_proj.bias.float().view(H, D)
        bias_pair = float(torch.sqrt(bias[h, d] ** 2 + bias[h, d + half] ** 2))
        Kb = K[None]
        err = {}
        for bits in (2, 4):
            for mode, dim in (("token", -1), ("channel", -2)):
                q = dequantize(quantize(Kb, bits, args.group, dim)).float()
                err[f"{mode}{bits}"] = float((q - Kb).norm() / Kb.norm())
        rows.append({"layer": layer, "outlier_ratio": float(pair.max() / pair.median()),
                     "top_pair": [h, d], "bias_share": bias_pair / float(pair[h, d]), "rel_error": err})

    md = [f"# Key statistics per layer: {args.model} ({args.tokens} WikiText-2 tokens, group {args.group})", "",
          "outlier = largest / median mean |k| over RoPE channel pairs; bias share = |k_proj bias| / mean |k| "
          "for that pair; errors = ||K - Q(K)|| / ||K||.", "",
          "| layer | outlier ratio | top pair (head, dim) | bias share | token-4 err | channel-4 err | token-2 err | channel-2 err |",
          "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        e = r["rel_error"]
        md.append(f"| {r['layer']} | {r['outlier_ratio']:.1f} | {tuple(r['top_pair'])} | {r['bias_share']:.2f} | "
                  f"{e['token4']:.3f} | {e['channel4']:.3f} | {e['token2']:.3f} | {e['channel2']:.3f} |")
    mean = {k: statistics.fmean(r["rel_error"][k] for r in rows) for k in rows[0]["rel_error"]}
    md.append(f"| mean | {statistics.fmean(r['outlier_ratio'] for r in rows):.1f} | | "
              f"{statistics.fmean(r['bias_share'] for r in rows):.2f} | {mean['token4']:.3f} | "
              f"{mean['channel4']:.3f} | {mean['token2']:.3f} | {mean['channel2']:.3f} |")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / (out.name + ".json")).write_text(json.dumps({"model": args.model, "rows": rows}, indent=2))
    (out.parent / (out.name + ".md")).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
