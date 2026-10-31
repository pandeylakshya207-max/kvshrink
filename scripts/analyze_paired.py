"""Paired comparison of every method against `full` on identical samples.

LongBench: per-sample score difference, per task and averaged over tasks (each
task weighted equally), optionally excluding floor-effect tasks.
Needle: per-sample correctness difference, per context length and pooled.
Cells: mean difference in points [95% stratified paired-bootstrap CI]; '*' marks
intervals that exclude 0.

Usage:
    python scripts/analyze_paired.py --longbench results/longbench_0.5b.json \
        --needle results/needle_0.5b.json --exclude passage_retrieval_en --output results/paired_0.5b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvshrink.stats import paired_bootstrap

REF = "full"


def pair_groups(samples: list[dict], method: str, stratum: str, key: tuple[str, ...],
                value: str, scale: float) -> dict[str, list[tuple[float, float]]]:
    """Group (ref, method) value pairs by `stratum`, matching samples on `key`."""
    ok = [s for s in samples if not s.get("oom")]
    ref = {tuple(s[k] for k in key): s for s in ok if s["method"] == REF}
    groups: dict[str, list[tuple[float, float]]] = {}
    for s in ok:
        if s["method"] != method:
            continue
        r = ref.get(tuple(s[k] for k in key))
        if r is not None:
            groups.setdefault(str(s[stratum]), []).append((scale * float(r[value]), scale * float(s[value])))
    return groups


def cell(stat: tuple[float, float, float]) -> str:
    p, lo, hi = stat
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]{star}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--longbench", default="results/longbench_0.5b.json")
    p.add_argument("--needle", default="results/needle_0.5b.json")
    p.add_argument("--exclude", nargs="*", default=["passage_retrieval_en"])
    p.add_argument("--output", default="results/paired_0.5b")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    out: dict = {"longbench": {}, "needle": {}}
    md = ["# Paired differences vs full (points; mean [95% paired-bootstrap CI]; * = CI excludes 0)", ""]

    lb = json.loads(Path(args.longbench).read_text())
    tasks = list(lb["config"]["tasks"])
    kept = [t for t in tasks if t not in args.exclude]
    md += [f"## LongBench ({lb['config']['n_per_task']} paired samples per task)", "",
           "| method | " + " | ".join(tasks) + " | avg (all) | avg (excl. " + ", ".join(args.exclude) + ") |",
           "|---|" + "---:|" * (len(tasks) + 2)]
    for m in lb["config"]["methods"]:
        if m == REF:
            continue
        g = pair_groups(lb["samples"], m, "task", ("task", "index"), "score", 100.0)
        row = {t: paired_bootstrap({t: g[t]}, args.n_boot) for t in tasks if t in g}
        row["avg_all"] = paired_bootstrap({t: g[t] for t in tasks if t in g}, args.n_boot)
        row["avg_kept"] = paired_bootstrap({t: g[t] for t in kept if t in g}, args.n_boot)
        out["longbench"][m] = row
        md.append(f"| {m} | " + " | ".join(cell(row[t]) if t in row else "-" for t in tasks)
                  + f" | {cell(row['avg_all'])} | {cell(row['avg_kept'])} |")

    nd = json.loads(Path(args.needle).read_text())
    ctxs = [str(c) for c in nd["contexts"]]
    n_per = len(nd["config"]["seeds"]) * len(nd["config"]["depths"])
    md += ["", f"## Needle-in-a-haystack ({n_per} paired samples per context)", "",
           "| method | " + " | ".join(ctxs) + " | pooled |", "|---|" + "---:|" * (len(ctxs) + 1)]
    for m in nd["config"]["methods"]:
        if m == REF:
            continue
        g = pair_groups(nd["samples"], m, "context", ("context", "seed", "depth"), "correct", 100.0)
        row = {c: paired_bootstrap({c: g[c]}, args.n_boot) for c in ctxs if c in g}
        row["pooled"] = paired_bootstrap({c: g[c] for c in ctxs if c in g}, args.n_boot)
        out["needle"][m] = row
        md.append(f"| {m} | " + " | ".join(cell(row[c]) if c in row else "-" for c in ctxs)
                  + f" | {cell(row['pooled'])} |")

    dest = Path(args.output)
    (dest.parent / (dest.name + ".json")).write_text(json.dumps(out, indent=2))
    (dest.parent / (dest.name + ".md")).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
