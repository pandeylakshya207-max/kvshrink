"""Needle-in-a-haystack (passkey retrieval) accuracy for each KV-cache method.

For each context length, depth and seed: a WikiText-2 *train* excerpt with one
passkey sentence inserted at that depth, followed by a question. The prompt is
prefilled in chunks of `prefill_chunk` tokens, identically for every method, so
later chunks attend to the compressed earlier ones. Then `gen_tokens` tokens are
decoded greedily. A sample is correct if the first number in the answer equals
the passkey. Prompts depend only on (seed, context), so all methods see the same ones.

Accuracy per (method, context) is the mean over depths within a seed; the table
reports the mean over seeds with a 95% t confidence interval.

Usage:
    python scripts/eval_needle.py --config configs/needle.yaml [--methods full kivi4] [--contexts 4096]
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvshrink.factory import make_cache
from kvshrink.model import QwenRunner
from kvshrink.needle import build_ids, extract_number, mean_ci95

DTYPES = {"float16": torch.float16, "float32": torch.float32}
FULL_KV_BYTES = None  # filled from the "full" method's measurement


def resolve_spec(spec: dict, ctx: int) -> dict:
    """Turn budget_frac / recent_frac into absolute token budgets for this context."""
    s = dict(spec)
    frac = s.pop("budget_frac", None)
    rfrac = s.pop("recent_frac", None)
    if frac is not None:
        s["budget"] = int(ctx * frac)
        if rfrac is not None:
            s["recent"] = int(s["budget"] * rfrac)
    return s


@torch.no_grad()
def run_batch(runner: QwenRunner, cache, ids: torch.Tensor, chunk: int, gen_tokens: int) -> torch.Tensor:
    """Chunked prefill of `ids` [B, T], then greedy decoding. Returns [B, gen_tokens]."""
    logits = None
    for s in range(0, ids.shape[1], chunk):
        logits = runner.forward(ids[:, s:s + chunk], cache, last_only=True)
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out = [nxt]
    for _ in range(gen_tokens - 1):
        nxt = runner.forward(nxt, cache, last_only=True)[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
    return torch.cat(out, dim=1)


def summarize(samples: list[dict], methods: list[str], contexts: list[int], depths: list[float]) -> dict:
    summary: dict = {}
    for m in methods:
        summary[m] = {}
        for ctx in contexts:
            rows = [s for s in samples if s["method"] == m and s["context"] == ctx]
            if not rows:
                continue
            if any(r.get("oom") for r in rows):
                summary[m][str(ctx)] = {"oom": True}
                continue
            seeds = sorted({r["seed"] for r in rows})
            per_seed = [statistics.fmean(r["correct"] for r in rows if r["seed"] == sd) for sd in seeds]
            acc, ci = mean_ci95(per_seed)
            summary[m][str(ctx)] = {
                "accuracy": acc,
                "ci95": ci,
                "per_seed": per_seed,
                "per_depth": {str(d): statistics.fmean(r["correct"] for r in rows if r["depth"] == d)
                              for d in depths},
                "kv_bytes_per_token": statistics.fmean(r["kv_bytes_per_token"] for r in rows),
                "n": len(rows),
            }
    return summary


def render_markdown(r: dict) -> str:
    cfg, env, summ = r["config"], r["env"], r["summary"]
    contexts, depths = r["contexts"], cfg["depths"]
    full = summ.get("full", {})
    lines = [
        f"# Needle-in-a-haystack (passkey retrieval): {r['model']} ({cfg['dtype']})",
        "",
        f"GPU: {env['gpu']} | torch {env['torch']} | transformers {env['transformers']} | generated {env['timestamp']}",
        "",
        f"Depths {depths}; seeds {cfg['seeds']}; chunked prefill of {cfg['prefill_chunk']} tokens for every "
        f"method; greedy decoding of {cfg['gen_tokens']} tokens. Cells: accuracy, mean over seeds "
        "+/- 95% t-CI (per-seed accuracy = mean over depths). In brackets: KV compression vs full.",
        "",
        "| method | spec | " + " | ".join(str(c) for c in contexts) + " |",
        "|---|---|" + "---:|" * len(contexts),
    ]
    for m, spec in cfg["methods"].items():
        if m not in summ:
            continue
        spec_s = ", ".join(f"{k}={v}" for k, v in spec.items())
        cells = []
        for c in contexts:
            s = summ[m].get(str(c))
            if s is None:
                cells.append("-")
            elif s.get("oom"):
                cells.append("OOM")
            else:
                f = full.get(str(c), {})
                ratio = f"{f['kv_bytes_per_token'] / s['kv_bytes_per_token']:.2f}x" if "kv_bytes_per_token" in f else "?"
                cells.append(f"{100 * s['accuracy']:.0f}% +/- {100 * s['ci95']:.0f} ({ratio})")
        lines.append(f"| {m} | {spec_s} | " + " | ".join(cells) + " |")
    lines += ["", "## Accuracy by needle depth (pooled over seeds)", "",
              "| method | context | " + " | ".join(str(d) for d in depths) + " |",
              "|---|---:|" + "---:|" * len(depths)]
    for m in summ:
        for c in contexts:
            s = summ[m].get(str(c))
            if s and not s.get("oom"):
                lines.append(f"| {m} | {c} | " + " | ".join(f"{100 * s['per_depth'][str(d)]:.0f}%" for d in depths) + " |")
    return "\n".join(lines) + "\n"


def write_results(results: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / (out.name + ".json")).write_text(json.dumps(results, indent=2))
    (out.parent / (out.name + ".md")).write_text(render_markdown(results))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/needle.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--contexts", nargs="*", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_name = args.model or cfg["model"]
    out = Path(args.output or cfg["output"])
    methods = {k: v for k, v in cfg["methods"].items() if not args.methods or k in args.methods}
    contexts = args.contexts or cfg["contexts"]
    depths = cfg["depths"]
    if not torch.cuda.is_available():
        raise SystemExit("eval_needle.py needs a CUDA GPU")

    dtype = DTYPES[cfg["dtype"]]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to("cuda").eval()
    runner = QwenRunner(model)

    template = tok.apply_chat_template([{"role": "user", "content": "\x00"}], tokenize=False,
                                       add_generation_prompt=True)
    head, tail = template.split("\x00")
    pre = tok(head + cfg["intro"], add_special_tokens=False).input_ids
    post = tok(cfg["question"] + tail, add_special_tokens=False).input_ids
    hs = cfg["haystack"]
    ds = load_dataset(hs["name"], hs["config"], split=hs["split"])
    source = tok("\n\n".join(ds["text"]), add_special_tokens=False).input_ids
    print(f"haystack source: {len(source)} tokens", flush=True)

    results: dict = {
        "model": model_name,
        "config": cfg,
        "contexts": contexts,
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "samples": [],
        "timing_s": {},
    }
    samples = results["samples"]

    for name, spec in methods.items():
        for ctx in contexts:
            rspec = resolve_spec(spec, ctx)
            t0 = time.perf_counter()
            try:
                for seed in cfg["seeds"]:
                    rng = random.Random(f"{seed}-{ctx}")
                    offset = rng.randrange(0, len(source) - ctx)
                    keys = [rng.randrange(10000, 100000) for _ in depths]
                    rows = []
                    for d, key in zip(depths, keys):
                        needle = tok(cfg["needle"].format(key=key), add_special_tokens=False).input_ids
                        rows.append(build_ids(pre, source, needle, post, ctx, d, offset))
                    ids = torch.tensor(rows, device="cuda")
                    cache = make_cache(rspec, model.config, len(rows), ctx + cfg["gen_tokens"], dtype, "cuda")
                    gen = run_batch(runner, cache, ids, cfg["prefill_chunk"], cfg["gen_tokens"])
                    kvb = cache.used_bytes() / (len(rows) * cache.tokens_seen(0))
                    del cache
                    for d, key, g in zip(depths, keys, gen.tolist()):
                        text = tok.decode(g, skip_special_tokens=True)
                        samples.append({
                            "method": name, "context": ctx, "seed": seed, "depth": d, "passkey": key,
                            "answer": text.strip()[:60], "correct": extract_number(text) == str(key),
                            "kv_bytes_per_token": kvb,
                        })
            except torch.cuda.OutOfMemoryError:
                samples[:] = [s for s in samples if not (s["method"] == name and s["context"] == ctx)]
                samples.append({"method": name, "context": ctx, "oom": True})
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            results["timing_s"][f"{name}@{ctx}"] = time.perf_counter() - t0
            s = summarize(samples, [name], [ctx], depths)[name].get(str(ctx), {})
            if s.get("oom"):
                print(f"{name} @ {ctx}: OOM", flush=True)
            else:
                print(f"{name} @ {ctx}: acc {100 * s['accuracy']:.0f}% +/- {100 * s['ci95']:.0f}, "
                      f"{s['kv_bytes_per_token']:.0f} B/token, {results['timing_s'][f'{name}@{ctx}']:.0f}s", flush=True)
        results["summary"] = summarize(samples, list(methods), contexts, depths)
        write_results(results, out)

    print(render_markdown(results))


if __name__ == "__main__":
    main()
