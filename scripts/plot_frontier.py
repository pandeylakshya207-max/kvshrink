"""Accuracy-vs-memory frontier: every method on three tasks, one panel each.

x = KV-cache memory as % of the fp16 cache (measured bytes/token, including
scales, zero points, fp16 windows and H2O scores). y = task quality. Each
method family is a line from the fp16 point through its compression levels.
Also writes the plotted points as CSV (the table view of the figure).

Usage:
    python scripts/plot_frontier.py --ppl results/ppl_0.5b.json --needle results/needle_0.5b.json \
        --longbench results/longbench_0.5b.json --exclude passage_retrieval_en --output results/frontier_0.5b
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette (slots 1-3, validated all-pairs) + text/grid tokens.
FAMILIES = {
    "quant": {"label": "Quantization (int8, KIVI)", "color": "#2a78d6", "marker": "o"},
    "streaming": {"label": "StreamingLLM (eviction)", "color": "#eb6834", "marker": "s"},
    "h2o": {"label": "H2O (eviction)", "color": "#1baf7a", "marker": "^"},
}
FULL_COLOR, TEXT, TEXT_2, GRID, SURFACE = "#52514e", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def family(spec: dict) -> str | None:
    t = spec["type"]
    return None if t == "full" else ("quant" if t in ("quant", "kivi") else t)


def short_name(spec: dict, window: int | None = None) -> str:
    t = spec["type"]
    if t == "full":
        return "fp16"
    if t == "quant":
        return f"int{spec['key_bits']}"
    if t == "kivi":
        return f"KIVI-{spec['bits']}"
    frac = spec.get("budget_frac") or (spec["budget"] / window if window else None)
    return f"{round(100 * frac)}%"


def ppl_points(path: Path) -> list[dict]:
    r = json.loads(path.read_text())
    window = r["config"]["window"]
    full = next(m for m in r["methods"].values() if m["spec"]["type"] == "full")
    return [{"method": k, "family": family(m["spec"]), "name": short_name(m["spec"], window),
             "mem_pct": 100 * m["kv_bytes_per_token"] / full["kv_bytes_per_token"], "y": m["ppl"]}
            for k, m in r["methods"].items()]


def needle_points(path: Path) -> list[dict]:
    r = json.loads(path.read_text())
    summ = r["summary"]
    ctxs = [str(c) for c in r["contexts"]]

    def kv(m: str) -> float:
        return statistics.fmean(summ[m][c]["kv_bytes_per_token"] for c in ctxs)

    return [{"method": m, "family": family(spec), "name": short_name(spec),
             "mem_pct": 100 * kv(m) / kv("full"),
             "y": 100 * statistics.fmean(summ[m][c]["accuracy"] for c in ctxs)}
            for m, spec in r["config"]["methods"].items() if m in summ]


def longbench_points(path: Path, exclude: list[str]) -> list[dict]:
    r = json.loads(path.read_text())
    summ = r["summary"]
    tasks = [t for t in r["config"]["tasks"] if t not in exclude]

    def kv(m: str) -> float:
        return statistics.fmean(summ[m]["tasks"][t]["kv_bytes_per_token"] for t in tasks)

    return [{"method": m, "family": family(spec), "name": short_name(spec),
             "mem_pct": 100 * kv(m) / kv("full"),
             "y": 100 * statistics.fmean(summ[m]["tasks"][t]["score"] for t in tasks)}
            for m, spec in r["config"]["methods"].items() if m in summ]


def draw(ax, points: list[dict], title: str, ylabel: str) -> None:
    full = next(p for p in points if p["family"] is None)
    for fam, style in FAMILIES.items():
        pts = sorted([p for p in points if p["family"] == fam], key=lambda p: -p["mem_pct"])
        if not pts:
            continue
        line = [full] + pts
        ax.plot([p["mem_pct"] for p in line], [p["y"] for p in line], color=style["color"],
                linewidth=1.5, zorder=2)
        ax.scatter([p["mem_pct"] for p in pts], [p["y"] for p in pts], s=56, marker=style["marker"],
                   color=style["color"], edgecolors=SURFACE, linewidths=1.5, zorder=3, label=style["label"])
    ax.scatter([full["mem_pct"]], [full["y"]], s=64, marker="D", color=FULL_COLOR,
               edgecolors=SURFACE, linewidths=1.5, zorder=4, label="fp16 cache (no compression)")
    ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.set_xlabel("KV-cache memory (% of fp16)", fontsize=9, color=TEXT_2)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_2)
    ax.set_xlim(0, 110)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.margins(y=0.12)


# Candidate label offsets (points): above, below, right, left, then diagonals.
OFFSETS = [(0, 10), (0, -12), (12, 0), (-12, 0), (10, 10), (-10, 10), (10, -12), (-10, -12),
           (0, 20), (0, -22), (18, 12), (-18, 12), (18, -14), (-18, -14)]


def place_labels(fig, ax, points: list[dict]) -> None:
    """Greedy label placement: first offset whose box overlaps no marker, line or placed label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    marks = [ax.transData.transform((q["mem_pct"], q["y"])) for q in points]
    lines = []  # sampled pixel points along every drawn line, so labels avoid crossing them
    for ln in ax.get_lines():
        xy = ax.transData.transform(ln.get_xydata())
        for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
            n = max(2, int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / 4))
            lines += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n + 1)]
    placed = []
    for q in sorted(points, key=lambda q: q["mem_pct"]):
        for dx, dy in OFFSETS:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            t = ax.annotate(q["name"], (q["mem_pct"], q["y"]), textcoords="offset points", xytext=(dx, dy),
                            ha=ha, va="center", fontsize=8, color=TEXT_2, zorder=5)
            bb = t.get_window_extent(renderer).expanded(1.08, 1.15)
            hit_mark = any(bb.x0 - 7 < mx < bb.x1 + 7 and bb.y0 - 7 < my < bb.y1 + 7 for mx, my in marks)
            hit_line = any(bb.x0 < lx < bb.x1 and bb.y0 < ly < bb.y1 for lx, ly in lines)
            if not hit_mark and not hit_line and not any(bb.overlaps(o) for o in placed):
                placed.append(bb)
                break
            t.remove()
        else:  # nothing fits: keep the first choice
            ax.annotate(q["name"], (q["mem_pct"], q["y"]), textcoords="offset points", xytext=OFFSETS[0],
                        ha="center", va="center", fontsize=8, color=TEXT_2, zorder=5)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ppl", default="results/ppl_0.5b.json")
    p.add_argument("--needle", default="results/needle_0.5b.json")
    p.add_argument("--longbench", default="results/longbench_0.5b.json")
    p.add_argument("--exclude", nargs="*", default=["passage_retrieval_en"])
    p.add_argument("--output", default="results/frontier_0.5b")
    p.add_argument("--model", default="Qwen2.5-0.5B-Instruct")
    args = p.parse_args()

    panels = [
        ("WikiText-2 perplexity (lower is better)", "Perplexity", ppl_points(Path(args.ppl))),
        ("Needle retrieval, 4k-16k (higher is better)", "Accuracy (%)", needle_points(Path(args.needle))),
        ("LongBench avg excl. " + ", ".join(args.exclude) + " (higher is better)", "Score (0-100)",
         longbench_points(Path(args.longbench), args.exclude)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE)
    for ax, (title, ylabel, pts) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        draw(ax, pts, title, ylabel)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.13, top=0.80, wspace=0.18)
    for ax, (_, _, pts) in zip(axes, panels):
        place_labels(fig, ax, pts)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(lab) for lab in [FAMILIES[f]["label"] for f in FAMILIES] + ["fp16 cache (no compression)"]]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=4,
               frameon=False, fontsize=9, labelcolor=TEXT_2, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f"KV-cache compression: quality vs memory ({args.model}, measured on a T4)",
                 fontsize=11, color=TEXT, y=0.995)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.parent / (out.name + ".png"), dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.parent / (out.name + ".svg"), bbox_inches="tight", facecolor=SURFACE)
    with open(out.parent / (out.name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "method", "family", "label", "kv_memory_pct_of_fp16", "value"])
        for title, _, pts in panels:
            for q in pts:
                w.writerow([title, q["method"], q["family"] or "fp16", q["name"], f"{q['mem_pct']:.1f}", f"{q['y']:.3f}"])
    print(f"wrote {out}.png, .svg, .csv")


if __name__ == "__main__":
    main()
