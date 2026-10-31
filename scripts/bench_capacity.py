"""Phase 5: concurrent-request capacity of each KV-cache method on one GPU.

For each method and context length, finds the largest batch whose cache (filled
synthetically to `context - decode_steps` tokens; see kvshrink.capacity) plus decode
steps fits in GPU memory: doubling, then binary search on OOM. Reports
  * MEASURED max batch with this implementation (non-fused attention: quantized
    caches dequantize one layer to fp16 per step; H2O builds explicit fp32 scores),
  * IDEAL max batch (ANALYTIC): (total memory - weights - reserve) / KV bytes per
    sequence, i.e. what a fused low-bit attention kernel could approach,
  * decode throughput at the measured max batch.

Usage:
    python scripts/bench_capacity.py --config configs/capacity.yaml [--methods full kivi4] [--contexts 4096]
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM

from kvshrink.capacity import fill_synthetic
from kvshrink.factory import make_cache, resolve_spec
from kvshrink.model import QwenRunner

GIB = 2**30
DTYPES = {"float16": torch.float16, "float32": torch.float32}


def free_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def cuda_time(fn: Callable[[], None]) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


def try_batch(runner: QwenRunner, spec: dict, batch: int, ctx: int, steps: int, vocab: int,
              dtype: torch.dtype) -> dict | None:
    """Decode `steps` tokens for `batch` sequences ending at `ctx` tokens. None on OOM."""
    cache = ids = None
    try:
        free_memory()
        torch.cuda.reset_peak_memory_stats()
        cache = make_cache(resolve_spec(spec, ctx), runner.cfg, batch, ctx, dtype, "cuda")
        fill_synthetic(cache, ctx - steps)
        ids = torch.randint(0, vocab, (batch, 1), device="cuda")
        runner.forward(ids, cache, last_only=True)  # warm-up step

        def run() -> None:
            for _ in range(steps - 1):
                runner.forward(ids, cache, last_only=True)

        t = cuda_time(run)
        return {"batch": batch, "decode_tok_s": batch * (steps - 1) / t,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                "kv_used_gib": cache.used_bytes() / GIB, "kv_bytes_per_seq": cache.used_bytes() / batch}
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del cache, ids
        free_memory()


def find_max_batch(runner, spec, ctx, steps, max_batch, vocab, dtype) -> dict:
    best, b = None, 1
    while b <= max_batch:
        res = try_batch(runner, spec, b, ctx, steps, vocab, dtype)
        if res is None:
            break
        best, b = res, b * 2
    if best is None:
        return {"max_batch": 0}
    lo, hi = best["batch"], min(b, max_batch + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        res = try_batch(runner, spec, mid, ctx, steps, vocab, dtype)
        if res is None:
            hi = mid
        else:
            lo, best = mid, res
    return {"max_batch": best["batch"], "hit_search_limit": best["batch"] >= max_batch, "at_max": best}


def render_markdown(r: dict) -> str:
    env, rows = r["env"], r["rows"]
    full = {row["context"]: row for row in rows if row["method"] == "full"}
    lines = [
        f"# Decode capacity per KV-cache method: {r['model']}",
        "",
        f"GPU: {env['gpu']} ({env['total_gib']:.1f} GiB) | weights {r['weights_gib']:.2f} GiB | torch {env['torch']} "
        f"| generated {env['timestamp']}",
        "",
        "Measured: largest batch that decodes without OOM, cache filled synthetically to the context length "
        "(contents are zeros; memory and compute do not depend on them). Ideal (ANALYTIC, not measured): "
        f"(total - weights - {r['config']['reserve_gib']} GiB reserve) / measured KV bytes per sequence.",
        "",
        "| method | context | measured max batch | vs full | ideal max batch (analytic) | vs full "
        "| decode tok/s at max batch | peak allocated GiB | KV GiB at max batch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        f = full.get(row["context"])
        if row["max_batch"] == 0:
            lines.append(f"| {row['method']} | {row['context']} | 0 | - | - | - | - | - | - |")
            continue
        m = row["at_max"]
        vs = f"{row['max_batch'] / f['max_batch']:.2f}x" if f and f["max_batch"] else "-"
        ivs = f"{row['ideal_max_batch'] / f['ideal_max_batch']:.2f}x" if f else "-"
        lim = " (limit)" if row.get("hit_search_limit") else ""
        lines.append(f"| {row['method']} | {row['context']} | {row['max_batch']}{lim} | {vs} | {row['ideal_max_batch']} "
                     f"| {ivs} | {m['decode_tok_s']:.0f} | {m['peak_allocated_gib']:.2f} | {m['kv_used_gib']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/capacity.yaml")
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--contexts", nargs="*", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    methods = {k: v for k, v in cfg["methods"].items() if not args.methods or k in args.methods}
    contexts = args.contexts or cfg["contexts"]
    out = Path(args.output or cfg["output"])
    if not torch.cuda.is_available():
        raise SystemExit("bench_capacity.py needs a CUDA GPU")

    dtype = DTYPES[cfg["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype).to("cuda").eval()
    runner = QwenRunner(model)
    vocab = model.config.vocab_size
    weights = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    budget = total - weights - cfg["reserve_gib"] * GIB
    results = {
        "model": cfg["model"], "config": cfg, "weights_gib": weights / GIB, "rows": [],
        "env": {"gpu": torch.cuda.get_device_name(0), "total_gib": total / GIB, "torch": torch.__version__,
                "transformers": transformers.__version__, "python": platform.python_version(),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }
    for name, spec in methods.items():
        for ctx in contexts:
            one = try_batch(runner, spec, 1, ctx, cfg["decode_steps"], vocab, dtype)
            row = find_max_batch(runner, spec, ctx, cfg["decode_steps"], cfg["max_batch"], vocab, dtype)
            row.update({"method": name, "context": ctx,
                        "ideal_max_batch": int(budget // one["kv_bytes_per_seq"]) if one else 0})
            results["rows"].append(row)
            print(f"{name} @ {ctx}: measured {row['max_batch']}, ideal {row['ideal_max_batch']}", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            (out.parent / (out.name + ".json")).write_text(json.dumps(results, indent=2))
            (out.parent / (out.name + ".md")).write_text(render_markdown(results))
    print(render_markdown(results))


if __name__ == "__main__":
    main()
