"""Pick the k least 2-bit-sensitive layers from a layer-sensitivity run and write held-out eval configs.

Sensitivity of layer i = PPL(KIVI-4 with layer i at 2 bits) - PPL(KIVI-4). The mixed
config puts the `k` least sensitive layers at 2 bits and the rest at 4 bits. It is
compared with K4V2 (similar memory) on data NOT used for the selection: the
WikiText-2 validation split and the needle test.

Usage: python scripts/make_mixed_config.py --sensitivity results/layer_sensitivity_0.5b.json --k 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

KIVI4 = {"type": "kivi", "bits": 4, "group_size": 32, "residual": 128}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensitivity", default="results/layer_sensitivity_0.5b.json")
    p.add_argument("--k", type=int, default=12)
    args = p.parse_args()

    r = json.loads(Path(args.sensitivity).read_text())["methods"]
    base = r["kivi4"]["ppl"]
    sens = {int(n[1:3]): m["ppl"] - base for n, m in r.items() if n.startswith("l") and n.endswith("_2bit")}
    chosen = sorted(sorted(sens, key=sens.get)[: args.k])
    print(f"{args.k} least sensitive layers: {chosen}")
    methods = {
        "full": {"type": "full"},
        "kivi4": dict(KIVI4),
        "kivi2": {**KIVI4, "bits": 2},
        "k4v2": {"type": "quant", "key_bits": 4, "value_bits": 2, "key_mode": "channel", "value_mode": "token",
                 "key_group": 32, "value_group": 32, "residual": 128},
        f"mixed{args.k}": {**KIVI4, "layer_overrides": {i: {"key_bits": 2, "value_bits": 2} for i in chosen}},
    }
    ppl = yaml.safe_load(Path("configs/ppl.yaml").read_text())
    ppl["dataset"]["split"] = "validation"
    ppl["methods"] = methods
    ppl["output"] = "results/mixed_ppl_val_0.5b"
    Path("configs/mixed_ppl.yaml").write_text(yaml.safe_dump(ppl, sort_keys=False))
    needle = yaml.safe_load(Path("configs/ablation_needle.yaml").read_text())
    needle["methods"] = methods
    needle["output"] = "results/mixed_needle_0.5b"
    Path("configs/mixed_needle.yaml").write_text(yaml.safe_dump(needle, sort_keys=False))
    print("wrote configs/mixed_ppl.yaml and configs/mixed_needle.yaml")


if __name__ == "__main__":
    main()
