"""Generate every results table in README.md from the JSON files in results/.

Writes the generated block between the markers
    <!-- BEGIN RESULTS -->  and  <!-- END RESULTS -->
in README.md, and also to results/summary_0.5b.md. No number in that block is typed
by hand: rerun an experiment, rerun this script, and the README follows.

Usage: python scripts/build_readme.py [--readme README.md] [--results results]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BEGIN, END = "<!-- BEGIN RESULTS -->", "<!-- END RESULTS -->"
MAIN = ["full", "int8", "kivi4", "kivi2", "stream50", "stream25", "h2o50", "h2o25"]
PPL_NAME = {"stream50": "stream1024", "stream25": "stream512", "h2o50": "h2o1024", "h2o25": "h2o512"}
LABEL = {
    "full": "fp16 (no compression)", "int8": "int8 per-token", "kivi4": "KIVI 4-bit", "kivi2": "KIVI 2-bit",
    "k4v2": "K4-V2 (4-bit keys, 2-bit values)", "stream50": "StreamingLLM, 50% budget",
    "stream25": "StreamingLLM, 25% budget", "h2o50": "H2O, 50% budget", "h2o25": "H2O, 25% budget",
}


def load(results: Path, name: str) -> dict | None:
    p = results / name
    return json.loads(p.read_text()) if p.exists() else None


def ci(t: list[float] | None) -> str:
    if not t:
        return "-"
    p, lo, hi = t
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]{star}"


def spec_str(spec: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in spec.items() if k != "layer_overrides") + (
        f", 2-bit layers={sorted(int(i) for i in spec['layer_overrides'])}" if "layer_overrides" in spec else "")


def spearman(a: list[float], b: list[float]) -> float:
    def rank(x: list[float]) -> list[int]:
        r = [0] * len(x)
        for k, i in enumerate(sorted(range(len(x)), key=lambda i: x[i])):
            r[i] = k
        return r
    ra, rb, n = rank(a), rank(b), len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def baseline_section(b: dict) -> list[str]:
    kv = b["kv_bytes_per_token"]
    lines = ["### Baseline (fp16 cache, batch 1)", "",
             f"KV cache: **{kv['measured_bytes_per_token']:.0f} bytes/token** measured with "
             f"`torch.cuda.memory_allocated` (analytic: {kv['analytic_bytes_per_token']:.0f}). "
             f"Weights: {b['weights_allocated_mib']:.0f} MiB.", "",
             "| context | prefill tok/s | decode tok/s |", "|---:|---:|---:|"]
    for r in b["latency"]:
        lines.append(f"| {r['context']} | {r['prefill_tok_s']['mean']:.0f} | {r['decode_tok_s']['mean']:.1f} |")
    return lines + [""]


def main_table(ppl: dict, needle: dict, lb: dict, paired: dict, cap: dict | None) -> list[str]:
    full_kv = statistics.fmean(needle["summary"]["full"][str(c)]["kv_bytes_per_token"] for c in needle["contexts"])
    tasks = list(lb["config"]["tasks"])
    kept = [t for t in tasks if t != "passage_retrieval_en"]
    capr = {(r["method"], r["context"]): r for r in cap["rows"]} if cap else {}
    lines = [
        "### Main comparison (Qwen2.5-0.5B-Instruct, T4)", "",
        "Compression = fp16 KV bytes / method KV bytes, measured on the needle runs (4k-16k mean; includes scales, "
        "zero points, fp16 windows, H2O scores). Paired deltas vs fp16 in points with 95% paired-bootstrap CI; "
        "`*` = CI excludes 0. LongBench average excludes passage_retrieval_en (fp16 itself scores near chance).", "",
        "| method | compression | WikiText-2 PPL (delta) | needle acc 4k-16k | needle delta | LongBench avg | LongBench delta "
        "| max concurrent @8k (vs fp16) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    full_cap = capr.get(("full", 8192))
    for m in MAIN:
        p = ppl["methods"][PPL_NAME.get(m, m)]
        kv = statistics.fmean(needle["summary"][m][str(c)]["kv_bytes_per_token"] for c in needle["contexts"])
        acc = 100 * statistics.fmean(needle["summary"][m][str(c)]["accuracy"] for c in needle["contexts"])
        lbs = 100 * statistics.fmean(lb["summary"][m]["tasks"][t]["score"] for t in kept)
        c = capr.get((m, 8192))
        capc = f"{c['max_batch']} ({c['max_batch'] / full_cap['max_batch']:.2f}x)" if c and full_cap else "-"
        nd = "-" if m == "full" else ci(paired["needle"][m]["pooled"])
        ld = "-" if m == "full" else ci(paired["longbench"][m]["avg_kept"])
        lines.append(f"| {LABEL[m]} | {full_kv / kv:.2f}x | {p['ppl']:.2f} ({p['ppl'] - ppl['methods']['full']['ppl']:+.2f}) "
                     f"| {acc:.0f}% | {nd} | {lbs:.1f} | {ld} | {capc} |")
    return lines + [""]


def ablation_table(title: str, note: str, ppl: dict, needle: dict, paired: dict) -> list[str]:
    ctx = str(needle["contexts"][0])
    full_ppl = ppl["methods"]["full"]["ppl"]
    lines = [f"### {title}", "", note, "",
             "| variant | spec | KV bytes/token | PPL (delta) | needle acc @8k | needle delta |",
             "|---|---|---:|---:|---:|---:|"]
    for m, p in ppl["methods"].items():
        if m == "full":
            continue
        s = needle["summary"].get(m, {}).get(ctx)
        acc = f"{100 * s['accuracy']:.0f}%" if s else "-"
        lines.append(f"| {m} | {spec_str(p['spec'])} | {p['kv_bytes_per_token']:.0f} | {p['ppl']:.2f} "
                     f"({p['ppl'] - full_ppl:+.2f}) | {acc} | {ci(paired['needle'].get(m, {}).get('pooled'))} |")
    return lines + [""]


def layer_section(sens: dict, keys: dict | None) -> list[str]:
    m = sens["methods"]
    base = m["kivi4"]["ppl"]
    d = {int(n[1:3]): v["ppl"] - base for n, v in m.items() if n.startswith("l") and n.endswith("_2bit")}
    order = sorted(d, key=d.get)
    fmt = lambda ls: ", ".join(f"{i} ({d[i]:+.3f})" for i in ls)  # noqa: E731
    lines = ["### Per-layer sensitivity (one layer at 2-bit, rest KIVI-4; WikiText-2 test PPL delta vs all-KIVI-4)", "",
             f"- Most sensitive: {fmt(order[::-1][:4])}",
             f"- Least sensitive: {fmt(order[:4])}",
             f"- Sum over the 12 least sensitive layers: {sum(d[i] for i in order[:12]):+.3f}; "
             f"over the other 12: {sum(d[i] for i in order[12:]):+.3f}"]
    if keys:
        rows = {r["layer"]: r for r in keys["rows"]}
        layers = sorted(d)
        rho = spearman([rows[i]["rel_error"]["channel2"] for i in layers], [d[i] for i in layers])
        mean = lambda k: statistics.fmean(r["rel_error"][k] for r in keys["rows"])  # noqa: E731
        lines += [f"- Spearman correlation between a layer's 2-bit per-channel key error and its sensitivity: {rho:.2f}",
                  f"- Mean relative key quantization error, per-token vs per-channel grouping: "
                  f"{mean('token4'):.3f} vs {mean('channel4'):.3f} (4-bit), {mean('token2'):.3f} vs {mean('channel2'):.3f} (2-bit)",
                  f"- Largest key outlier ratio (max / median channel-pair magnitude): "
                  f"{max(r['outlier_ratio'] for r in keys['rows']):.1f} (layer "
                  f"{max(keys['rows'], key=lambda r: r['outlier_ratio'])['layer']}); see results/key_stats_0.5b.md"]
    return lines + [""]


def capacity_section(cap: dict) -> list[str]:
    full = {r["context"]: r for r in cap["rows"] if r["method"] == "full"}
    lines = ["### Capacity: concurrent sequences on one T4", "",
             "Measured = largest batch that decodes without OOM (cache synthetically filled to the context length). "
             "Ideal = ANALYTIC (free memory after weights / KV bytes per sequence), i.e. what a fused low-bit "
             "attention kernel could approach. Throughput is aggregate decode tokens/s at the measured max batch.", "",
             "| method | context | measured max batch (vs fp16) | ideal max batch (analytic) | decode tok/s at max batch |",
             "|---|---:|---:|---:|---:|"]
    for r in cap["rows"]:
        if not r["max_batch"]:
            continue
        f = full[r["context"]]["max_batch"]
        lines.append(f"| {r['method']} | {r['context']} | {r['max_batch']} ({r['max_batch'] / f:.2f}x) | "
                     f"{r['ideal_max_batch']} | {r['at_max']['decode_tok_s']:.0f} |")
    return lines + [""]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--readme", default="README.md")
    p.add_argument("--results", default="results")
    args = p.parse_args()
    R = Path(args.results)

    out = ["_Generated by `python scripts/build_readme.py` from `results/*.json`; do not edit by hand._", ""]
    if (b := load(R, "baseline.json")):
        out += baseline_section(b)
    out += main_table(load(R, "ppl_0.5b.json"), load(R, "needle_0.5b.json"), load(R, "longbench_0.5b.json"),
                      load(R, "paired_0.5b.json"), load(R, "capacity_0.5b.json"))
    out += ablation_table(
        "Ablations around KIVI-2 (WikiText-2 test PPL; needle at 8k, 30 paired samples)",
        "One factor changed at a time: fp16 window (r), group size (g), key/value bit split, per-token vs per-channel keys.",
        load(R, "ablation_ppl_0.5b.json"), load(R, "ablation_needle_0.5b.json"), load(R, "paired_ablation_0.5b.json"))
    out += layer_section(load(R, "layer_sensitivity_0.5b.json"), load(R, "key_stats_0.5b.json"))
    out += ablation_table(
        "Held-out check: layer-wise mixed precision vs K4-V2 at equal memory",
        "Layers for `mixed12` were chosen from the WikiText-2 *test* sensitivity sweep, so it is evaluated on the "
        "*validation* split and on the needle test.",
        load(R, "mixed_ppl_val_0.5b.json"), load(R, "mixed_needle_0.5b.json"), load(R, "paired_mixed_0.5b.json"))
    if (cap := load(R, "capacity_0.5b.json")):
        out += capacity_section(cap)
    block = "\n".join(out).rstrip() + "\n"

    (R / "summary_0.5b.md").write_text(block)
    readme = Path(args.readme)
    text = readme.read_text() if readme.exists() else f"{BEGIN}\n{END}\n"
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{readme} has no {BEGIN} / {END} markers")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{BEGIN}\n{block}{END}{tail}")
    print(f"updated {readme} and {R / 'summary_0.5b.md'} ({len(block.splitlines())} lines)")


if __name__ == "__main__":
    main()
