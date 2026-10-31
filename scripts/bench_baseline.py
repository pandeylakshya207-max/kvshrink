"""Phase 1 baseline benchmark on GPU (full-precision StaticKVCache).

Measures:
  1. KV-cache bytes per token, from torch.cuda.memory_allocated deltas.
  2. Prefill and decode throughput at batch size 1 for several context lengths.
  3. Decode-phase capacity: the largest batch whose KV cache + decode step fits on
     the GPU at a given context length. The cache is filled SYNTHETICALLY (zeros,
     lengths set directly), so prefill memory is NOT included. This models a server
     that prefills requests separately and decodes them together.

Writes <output>.json (raw numbers + environment) and <output>.md (tables).

Usage:
    python scripts/bench_baseline.py --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM

from kvshrink.model import QwenRunner

MIB = 2**20
DTYPES = {"float16": torch.float16, "float32": torch.float32}


def free_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def nvidia_smi_used_mib() -> int | None:
    """Device-wide used memory as nvidia-smi reports it (all processes + CUDA context)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def cuda_time(fn: Callable[[], None]) -> float:
    """GPU time of `fn` in seconds, measured with CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


def mean_std(xs: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0}


def measure_kv_bytes_per_token(runner: QwenRunner, n_tokens: int) -> dict:
    """Allocate a batch-1 cache of `n_tokens` and measure the allocator delta."""
    free_memory()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    cache = runner.new_cache(1, n_tokens)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    result = {
        "n_tokens": n_tokens,
        "measured_bytes_per_token": (after - before) / n_tokens,
        "analytic_bytes_per_token": cache.memory_bytes() / n_tokens,
    }
    del cache
    free_memory()
    return result


def bench_latency(
    runner: QwenRunner, ctx: int, decode_tokens: int, warmup: int, repeats: int,
    vocab: int, gen: torch.Generator,
) -> dict:
    """Batch-1 prefill of `ctx` random tokens, then `decode_tokens` greedy steps."""
    ids = torch.randint(0, vocab, (1, ctx), generator=gen).to(runner.device)
    prefill_tps: list[float] = []
    decode_tps: list[float] = []
    peaks: list[float] = []
    reserved = smi = None
    for r in range(warmup + repeats):
        free_memory()
        torch.cuda.reset_peak_memory_stats()
        cache = runner.new_cache(1, ctx + decode_tokens)
        state: dict[str, torch.Tensor] = {}

        def prefill() -> None:
            logits = runner.forward(ids, cache, last_only=True)
            state["next"] = logits[:, -1].argmax(-1, keepdim=True)

        def decode() -> None:
            nxt = state["next"]
            for _ in range(decode_tokens):
                nxt = runner.forward(nxt, cache, last_only=True)[:, -1].argmax(-1, keepdim=True)

        t_prefill = cuda_time(prefill)
        t_decode = cuda_time(decode)
        if r >= warmup:
            prefill_tps.append(ctx / t_prefill)
            decode_tps.append(decode_tokens / t_decode)
            peaks.append(torch.cuda.max_memory_allocated() / MIB)
            reserved = torch.cuda.memory_reserved() / MIB
            smi = nvidia_smi_used_mib()
        del cache, state
    free_memory()
    return {
        "context": ctx,
        "decode_tokens": decode_tokens,
        "prefill_tok_s": mean_std(prefill_tps),
        "decode_tok_s": mean_std(decode_tps),
        "decode_ms_per_token": mean_std([1000.0 / x for x in decode_tps]),
        "peak_allocated_mib": max(peaks),
        "reserved_mib": reserved,
        "nvidia_smi_used_mib": smi,
    }


def try_decode_batch(runner: QwenRunner, batch: int, ctx: int, steps: int, vocab: int) -> dict | None:
    """Decode `steps` tokens for `batch` sequences ending at `ctx` tokens. None on OOM."""
    cache = ids = None
    try:
        free_memory()
        torch.cuda.reset_peak_memory_stats()
        cache = runner.new_cache(batch, ctx)
        cache.lengths = [ctx - steps] * cache.num_layers  # synthetic fill
        ids = torch.randint(0, vocab, (batch, 1), device=runner.device)
        runner.forward(ids, cache, last_only=True)  # warm-up step

        def run() -> None:
            for _ in range(steps - 1):
                runner.forward(ids, cache, last_only=True)

        t = cuda_time(run)
        return {
            "batch": batch,
            "decode_tok_s": batch * (steps - 1) / t,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
            "kv_cache_mib": cache.memory_bytes() / MIB,
        }
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del cache, ids
        free_memory()


def find_max_batch(runner: QwenRunner, ctx: int, steps: int, max_batch: int, vocab: int) -> dict:
    """Double the batch until OOM, then binary-search the boundary."""
    best, b = None, 1
    while b <= max_batch:
        res = try_decode_batch(runner, b, ctx, steps, vocab)
        if res is None:
            break
        best, b = res, b * 2
    if best is None:
        return {"context": ctx, "max_batch": 0}
    lo, hi = best["batch"], min(b, max_batch + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        res = try_decode_batch(runner, mid, ctx, steps, vocab)
        if res is None:
            hi = mid
        else:
            lo, best = mid, res
    return {
        "context": ctx,
        "max_batch": best["batch"],
        "hit_search_limit": best["batch"] >= max_batch,
        "at_max_batch": best,
    }


def fmt(ms: dict[str, float]) -> str:
    return f"{ms['mean']:.1f} +/- {ms['std']:.1f}"


def render_markdown(r: dict) -> str:
    kv, env, cfg = r["kv_bytes_per_token"], r["env"], r["config"]
    lines = [
        f"# Baseline: {r['model']} ({cfg['dtype']}, full-precision KV cache)",
        "",
        f"GPU: {env['gpu']} | torch {env['torch']} (CUDA {env['cuda']}) | "
        f"transformers {env['transformers']} | generated {env['timestamp']}",
        "",
        "## KV cache size",
        "",
        f"- Measured (torch.cuda.memory_allocated delta over {kv['n_tokens']} tokens): "
        f"{kv['measured_bytes_per_token']:.0f} bytes/token",
        f"- Analytic (2 x layers x kv_heads x head_dim x bytes): {kv['analytic_bytes_per_token']:.0f} bytes/token",
        f"- Model weights allocated: {r['weights_allocated_mib']:.0f} MiB",
        "",
        f"## Batch-1 latency (random token ids; mean +/- std over {cfg['latency']['repeats']} runs)",
        "",
        "| context | prefill tok/s | decode tok/s | decode ms/token | peak allocated MiB | reserved MiB | nvidia-smi MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in r["latency"]:
        lines.append(
            f"| {row['context']} | {fmt(row['prefill_tok_s'])} | {fmt(row['decode_tok_s'])} | "
            f"{fmt(row['decode_ms_per_token'])} | {row['peak_allocated_mib']:.0f} | "
            f"{row['reserved_mib']:.0f} | {row['nvidia_smi_used_mib']} |"
        )
    lines += [
        "",
        "## Decode capacity (SYNTHETIC cache fill; prefill memory not included)",
        "",
        "| context | max batch | decode tok/s at max batch | peak allocated MiB | KV cache MiB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in r["capacity"]:
        if row["max_batch"] == 0:
            lines.append(f"| {row['context']} | 0 | - | - | - |")
            continue
        m = row["at_max_batch"]
        limit = " (search limit)" if row["hit_search_limit"] else ""
        lines.append(
            f"| {row['context']} | {row['max_batch']}{limit} | {m['decode_tok_s']:.0f} | "
            f"{m['peak_allocated_mib']:.0f} | {m['kv_cache_mib']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--model", default=None, help="override the model in the config")
    p.add_argument("--output", default=None, help="output path without extension")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_name = args.model or cfg["model"]
    out = Path(args.output or cfg["output"])
    if not torch.cuda.is_available():
        raise SystemExit("bench_baseline.py needs a CUDA GPU")

    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    smi_idle = nvidia_smi_used_mib()
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPES[cfg["dtype"]]).to("cuda").eval()
    runner = QwenRunner(model)
    vocab = model.config.vocab_size

    results: dict = {
        "model": model_name,
        "config": cfg,
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "nvidia_smi_before_load_mib": smi_idle,
        "weights_allocated_mib": torch.cuda.memory_allocated() / MIB,
    }

    results["kv_bytes_per_token"] = measure_kv_bytes_per_token(runner, cfg["kv_probe_tokens"])
    print("kv bytes/token:", results["kv_bytes_per_token"], flush=True)

    lat = cfg["latency"]
    results["latency"] = []
    for ctx in lat["context_lengths"]:
        row = bench_latency(runner, ctx, lat["decode_tokens"], lat["warmup"], lat["repeats"], vocab, gen)
        results["latency"].append(row)
        print(f"latency ctx={ctx}: prefill {fmt(row['prefill_tok_s'])} tok/s, "
              f"decode {fmt(row['decode_tok_s'])} tok/s", flush=True)

    cap = cfg["capacity"]
    results["capacity"] = []
    for ctx in cap["context_lengths"]:
        row = find_max_batch(runner, ctx, cap["decode_steps"], cap["max_batch"], vocab)
        results["capacity"].append(row)
        print(f"capacity ctx={ctx}: max batch {row['max_batch']}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.parent / (out.name + ".json").write_text(json.dumps(results, indent=2))
    md = render_markdown(results)
    out.parent / (out.name + ".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
