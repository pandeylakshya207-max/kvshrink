"""LongBench subset: task scores for each KV-cache method.

Each prompt is built LongBench-style: the task template is filled, the prompt is
truncated in the middle to `max_input_tokens`, and the chat template is applied.
Prefill runs in chunks of `prefill_chunk` tokens (same for every method), then
greedy decoding until EOS or the task's max_gen. Eviction budgets given as
fractions are resolved per prompt. Every method sees the same seeded sample of
`n_per_task` examples per task. Scores are 0-100 with 95% bootstrap CIs.

Results are written after each (method, task); rerunning skips finished pairs.

Usage:
    python scripts/eval_longbench.py --config configs/longbench.yaml [--methods full kivi4] [--tasks qasper]
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import statistics
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import yaml
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvshrink.factory import make_cache, resolve_spec
from kvshrink.longbench import METRICS, bootstrap_ci, middle_truncate
from kvshrink.model import QwenRunner

DTYPES = {"float16": torch.float16, "float32": torch.float32}


def load_task(task: str) -> list[dict]:
    """LongBench test split for `task`; falls back to the repo's data.zip."""
    try:
        return [dict(r) for r in load_dataset("THUDM/LongBench", task, split="test")]
    except Exception as e:  # noqa: BLE001 - any loader failure -> fallback
        print(f"load_dataset failed for {task} ({type(e).__name__}); using data.zip", flush=True)
        path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
        with zipfile.ZipFile(path) as z, z.open(f"data/{task}.jsonl") as f:
            return [json.loads(line) for line in f]


@torch.no_grad()
def generate(runner: QwenRunner, cache, ids: torch.Tensor, chunk: int, max_new: int, stop: set[int]) -> list[int]:
    """Chunked prefill of ids [1, T], then greedy decoding until a stop token or max_new."""
    logits = None
    for s in range(0, ids.shape[1], chunk):
        logits = runner.forward(ids[:, s:s + chunk], cache, last_only=True)
    out: list[int] = []
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(max_new):
        t = int(nxt.item())
        if t in stop:
            break
        out.append(t)
        nxt = runner.forward(nxt, cache, last_only=True)[:, -1].argmax(-1, keepdim=True)
    return out


def summarize(samples: list[dict], methods: list[str], tasks: list[str]) -> dict:
    summ: dict = {}
    for m in methods:
        summ[m] = {"tasks": {}}
        means = []
        for t in tasks:
            rows = [s for s in samples if s["method"] == m and s["task"] == t]
            if not rows:
                continue
            if any(r.get("oom") for r in rows):
                summ[m]["tasks"][t] = {"oom": True}
                continue
            mean, lo, hi = bootstrap_ci([r["score"] for r in rows])
            summ[m]["tasks"][t] = {"score": mean, "ci_lo": lo, "ci_hi": hi, "n": len(rows),
                                   "kv_bytes_per_token": statistics.fmean(r["kv_bytes_per_token"] for r in rows)}
            means.append(mean)
        complete = [t for t in tasks if "score" in summ[m]["tasks"].get(t, {})]
        if complete and len(complete) == len(tasks):
            summ[m]["average"] = statistics.fmean(means)
            summ[m]["kv_bytes_per_token"] = statistics.fmean(
                summ[m]["tasks"][t]["kv_bytes_per_token"] for t in tasks)
    return summ


def render_markdown(r: dict) -> str:
    cfg, env, summ = r["config"], r["env"], r["summary"]
    tasks = list(cfg["tasks"])
    full_kv = summ.get("full", {}).get("kv_bytes_per_token")
    lines = [
        f"# LongBench subset: {r['model']} ({cfg['dtype']})",
        "",
        f"GPU: {env['gpu']} | torch {env['torch']} | transformers {env['transformers']} | generated {env['timestamp']}",
        "",
        f"{cfg['n_per_task']} examples per task (seeded sample, same for all methods); prompts middle-truncated "
        f"to {cfg['max_input_tokens']} tokens; chunked prefill of {cfg['prefill_chunk']}; greedy decoding. "
        "Scores 0-100, cells: mean [95% bootstrap CI].",
        "",
        "| method | " + " | ".join(tasks) + " | average | compression vs full |",
        "|---|" + "---:|" * (len(tasks) + 2),
    ]
    for m in cfg["methods"]:
        if m not in summ:
            continue
        cells = []
        for t in tasks:
            s = summ[m]["tasks"].get(t)
            if s is None:
                cells.append("-")
            elif s.get("oom"):
                cells.append("OOM")
            else:
                cells.append(f"{100 * s['score']:.1f} [{100 * s['ci_lo']:.1f}, {100 * s['ci_hi']:.1f}]")
        avg = f"{100 * summ[m]['average']:.1f}" if "average" in summ[m] else "-"
        kv = summ[m].get("kv_bytes_per_token")
        ratio = f"{full_kv / kv:.2f}x" if full_kv and kv else "-"
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {avg} | {ratio} |")
    return "\n".join(lines) + "\n"


def write_results(results: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / (out.name + ".json")).write_text(json.dumps(results, indent=2))
    (out.parent / (out.name + ".md")).write_text(render_markdown(results))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/longbench.yaml")
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.output or cfg["output"])
    methods = {k: v for k, v in cfg["methods"].items() if not args.methods or k in args.methods}
    tasks = {k: v for k, v in cfg["tasks"].items() if not args.tasks or k in args.tasks}
    if not torch.cuda.is_available():
        raise SystemExit("eval_longbench.py needs a CUDA GPU")

    dtype = DTYPES[cfg["dtype"]]
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype).to("cuda").eval()
    runner = QwenRunner(model)
    stop = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")) if i is not None}

    json_path = out.parent / (out.name + ".json")
    if json_path.exists():
        results = json.loads(json_path.read_text())
        results["config"] = cfg
        print(f"resuming: {len(results['samples'])} samples already in {json_path}", flush=True)
    else:
        results = {"model": cfg["model"], "config": cfg, "samples": [], "timing_s": {}}
    results["env"] = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    samples = results["samples"]
    done = {(s["method"], s["task"]) for s in samples}

    for task, tcfg in tasks.items():
        if all((m, task) in done for m in methods):
            continue
        data = load_task(task)
        idx = sorted(random.Random(cfg["seed"]).sample(range(len(data)), min(cfg["n_per_task"], len(data))))
        prompts = []
        n_trunc = 0
        for i in idx:
            ex = data[i]
            raw = tok(tcfg["prompt"].format(context=ex["context"], input=ex["input"]), add_special_tokens=False).input_ids
            n_trunc += len(raw) > cfg["max_input_tokens"]
            text = tok.decode(middle_truncate(raw, cfg["max_input_tokens"]), skip_special_tokens=True)
            chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            prompts.append((i, tok(chat, add_special_tokens=False).input_ids, ex["answers"]))
        lens = [len(p[1]) for p in prompts]
        print(f"{task}: {len(prompts)} prompts, {n_trunc} truncated, tokens min/mean/max "
              f"{min(lens)}/{statistics.fmean(lens):.0f}/{max(lens)}", flush=True)

        for name, spec in methods.items():
            if (name, task) in done:
                continue
            t0 = time.perf_counter()
            rows = []
            for i, ids, answers in prompts:
                try:
                    cache = make_cache(resolve_spec(spec, len(ids)), model.config, 1,
                                       len(ids) + tcfg["max_gen"], dtype, "cuda")
                    gen = generate(runner, cache, torch.tensor([ids], device="cuda"),
                                   cfg["prefill_chunk"], tcfg["max_gen"], stop)
                    kvb = cache.used_bytes() / cache.tokens_seen(0)
                    del cache
                    pred = tok.decode(gen, skip_special_tokens=True)
                    if tcfg.get("first_line"):
                        pred = pred.lstrip("\n").split("\n")[0]
                    rows.append({"method": name, "task": task, "index": i, "prompt_tokens": len(ids),
                                 "prediction": pred[:300], "answers": answers,
                                 "score": METRICS[tcfg["metric"]](pred, answers), "kv_bytes_per_token": kvb})
                except torch.cuda.OutOfMemoryError:
                    rows = [{"method": name, "task": task, "oom": True}]
                    break
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
            samples.extend(rows)
            results["timing_s"][f"{name}@{task}"] = time.perf_counter() - t0
            results["summary"] = summarize(samples, list(cfg["methods"]), list(cfg["tasks"]))
            write_results(results, out)
            s = results["summary"][name]["tasks"].get(task, {})
            msg = "OOM" if s.get("oom") else f"{100 * s['score']:.1f} [{100 * s['ci_lo']:.1f}, {100 * s['ci_hi']:.1f}]"
            print(f"{name} @ {task}: {msg}, {time.perf_counter() - t0:.0f}s", flush=True)

    print(render_markdown(results))


if __name__ == "__main__":
    main()
