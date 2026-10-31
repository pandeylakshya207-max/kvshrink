"""WikiText-2 perplexity for each KV-cache method, with decode-style chunked scoring.

For each window of `window` tokens: prefill the first `prefix` tokens, then score
the remaining tokens in chunks of `chunk`. Every scored chunk attends to the past
as the cache stored it (compressed) and to itself in full precision, so compression
error reaches every scored token. chunk=1 is exact token-by-token decoding.

Writes <output>.json and <output>.md.

Usage:
    python scripts/eval_ppl.py --config configs/ppl.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvshrink.factory import make_cache
from kvshrink.model import QwenRunner

MIB = 2**20
DTYPES = {"float16": torch.float16, "float32": torch.float32}


def load_windows(tok, ds_cfg: dict, window: int, num_windows: int) -> torch.Tensor:
    """Tokenize the split (documents joined by blank lines) into [n, window] windows."""
    ds = load_dataset(ds_cfg["name"], ds_cfg["config"], split=ds_cfg["split"])
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = min(num_windows, ids.numel() // window)
    return ids[: n * window].view(n, window)


@torch.no_grad()
def score_batch(runner: QwenRunner, cache, ids: torch.Tensor, prefix: int, chunk: int) -> tuple[float, int]:
    """Sum of NLL (nats) and token count for positions prefix..T-1 of `ids` [B, T]."""
    T = ids.shape[1]
    prev = runner.forward(ids[:, :prefix], cache, last_only=True)[:, -1]
    nll = torch.zeros((), dtype=torch.float64, device=ids.device)
    n = 0
    for s in range(prefix, T, chunk):
        e = min(s + chunk, T)
        out = runner.forward(ids[:, s:e], cache)
        pred = torch.cat([prev[:, None], out[:, :-1]], dim=1).float()
        nll += F.cross_entropy(pred.flatten(0, 1), ids[:, s:e].flatten(), reduction="sum").double()
        n += ids[:, s:e].numel()
        prev = out[:, -1]
    return nll.item(), n


def render_markdown(r: dict) -> str:
    cfg, env = r["config"], r["env"]
    lines = [
        f"# WikiText-2 perplexity: {r['model']} ({cfg['dtype']})",
        "",
        f"GPU: {env['gpu']} | torch {env['torch']} | transformers {env['transformers']} | "
        f"generated {env['timestamp']}",
        "",
        f"{r['n_windows']} windows x {cfg['window']} tokens; prefix {cfg['prefix']} tokens prefilled, "
        f"remaining {r['scored_tokens_per_window']} per window scored in chunks of {cfg['chunk']} "
        f"({r['n_windows'] * r['scored_tokens_per_window']} scored tokens). Deterministic: no sampling.",
        "",
        "| method | spec | PPL | delta PPL vs full | KV bytes/token | compression vs full | peak allocated MiB | time s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in r["methods"].items():
        spec = ", ".join(f"{k}={v}" for k, v in m["spec"].items())
        lines.append(
            f"| {name} | {spec} | {m['ppl']:.3f} | {m.get('delta_ppl', float('nan')):+.3f} | "
            f"{m['kv_bytes_per_token']:.0f} | {m.get('compression', float('nan')):.2f}x | "
            f"{m['peak_allocated_mib']:.0f} | {m['seconds']:.0f} |"
        )
    lines += ["", "KV bytes/token = cache.used_bytes() / (batch x window), averaged over batches; "
              "includes scales, zero points and the fp16 residual window."]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ppl.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--methods", nargs="*", default=None, help="subset of method names to run")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_name = args.model or cfg["model"]
    out = Path(args.output or cfg["output"])
    methods = {k: v for k, v in cfg["methods"].items() if not args.methods or k in args.methods}
    if not torch.cuda.is_available():
        raise SystemExit("eval_ppl.py needs a CUDA GPU")

    dtype = DTYPES[cfg["dtype"]]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to("cuda").eval()
    runner = QwenRunner(model)
    windows = load_windows(tok, cfg["dataset"], cfg["window"], cfg["num_windows"])
    print(f"{windows.shape[0]} windows of {cfg['window']} tokens", flush=True)

    results: dict = {
        "model": model_name,
        "config": cfg,
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "n_windows": windows.shape[0],
        "scored_tokens_per_window": cfg["window"] - cfg["prefix"],
        "methods": {},
    }

    bs = cfg["batch_size"]
    for name, spec in methods.items():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        nll, n, kv = 0.0, 0, []
        for b0 in range(0, windows.shape[0], bs):
            ids = windows[b0:b0 + bs].to("cuda")
            cache = make_cache(spec, model.config, ids.shape[0], cfg["window"], dtype, "cuda")
            s, c = score_batch(runner, cache, ids, cfg["prefix"], cfg["chunk"])
            nll += s
            n += c
            kv.append(cache.used_bytes() / (ids.shape[0] * cfg["window"]))
            del cache
        torch.cuda.synchronize()
        res = {
            "spec": spec,
            "ppl": math.exp(nll / n),
            "nll_per_token": nll / n,
            "scored_tokens": n,
            "kv_bytes_per_token": statistics.fmean(kv),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
            "seconds": time.perf_counter() - t0,
        }
        results["methods"][name] = res
        print(f"{name}: ppl {res['ppl']:.3f}, {res['kv_bytes_per_token']:.0f} B/token, "
              f"{res['seconds']:.0f}s", flush=True)

    full = next((m for m in results["methods"].values() if m["spec"]["type"] == "full"), None)
    if full:
        for m in results["methods"].values():
            m["delta_ppl"] = m["ppl"] - full["ppl"]
            m["compression"] = full["kv_bytes_per_token"] / m["kv_bytes_per_token"]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.parent / (out.name + ".json").write_text(json.dumps(results, indent=2))
    md = render_markdown(results)
    out.parent / (out.name + ".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
