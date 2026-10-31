"""Generate the per-layer sensitivity config: KIVI-4 everywhere, one layer at a time at 2 bits.

Usage: python scripts/make_layer_config.py --layers 24 --output configs/layer_sensitivity.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

BASE = {"type": "kivi", "bits": 4, "group_size": 32, "residual": 128}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", default="configs/ppl.yaml")
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--output", default="configs/layer_sensitivity.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.base_config).read_text())
    methods: dict = {"full": {"type": "full"}, "kivi4": dict(BASE), "kivi2": {**BASE, "bits": 2}}
    for i in range(args.layers):
        methods[f"l{i:02d}_2bit"] = {**BASE, "layer_overrides": {i: {"key_bits": 2, "value_bits": 2}}}
    cfg["methods"] = methods
    cfg["output"] = "results/layer_sensitivity_0.5b"
    Path(args.output).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {args.output} with {len(methods)} methods")


if __name__ == "__main__":
    main()
