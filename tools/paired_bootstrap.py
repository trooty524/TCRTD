"""Paired comparison of two methods over seeds, with identity-block bootstrap intervals.

    python tools/paired_bootstrap.py \
        --baseline runs/market_b14_m0_s{13,42,2026}/test \
        --method   runs/market_b14_tcrtd_s{13,42,2026}/test

Each directory holds the per_query.npz written by evaluate.py; the i-th
baseline and method directories must share a seed. Reports the mean mAP of
both methods, the seed-level differences, and the 95% interval of the
seed-averaged per-query AP difference (10,000 identity-block replicates).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcrtd.metrics import identity_block_interval  # noqa: E402


def load(directory: Path) -> dict:
    data = np.load(directory / "per_query.npz")
    return {name: data[name] for name in data.files}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", nargs="+", type=Path, required=True)
    p.add_argument("--method", nargs="+", type=Path, required=True)
    args = p.parse_args(argv)
    if len(args.baseline) != len(args.method):
        p.error("give one baseline and one method directory per seed")

    base = [load(d) for d in args.baseline]
    meth = [load(d) for d in args.method]
    for b, m in zip(base, meth):
        if not np.array_equal(b["keys"], m["keys"]):
            raise ValueError("baseline and method were evaluated on different queries")

    seed_map = [(100 * np.nanmean(b["ap"]), 100 * np.nanmean(m["ap"])) for b, m in zip(base, meth)]
    delta = np.mean([100 * (m["ap"] - b["ap"]) for b, m in zip(base, meth)], axis=0)
    valid = np.isfinite(delta)
    low, high = identity_block_interval(delta[valid], base[0]["identities"][valid])
    report = {
        "baseline_mAP": float(np.mean([s[0] for s in seed_map])),
        "method_mAP": float(np.mean([s[1] for s in seed_map])),
        "delta_mAP": float(np.mean([s[1] - s[0] for s in seed_map])),
        "seed_deltas": [round(float(s[1] - s[0]), 2) for s in seed_map],
        "interval_95": [round(low, 2), round(high, 2)],
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
