"""Run fixed k-reciprocal re-ranking on saved test descriptors.

    python evaluate.py ... --save-features --output runs/x/test
    python tools/rerank.py --features runs/x/test/features.npz \
        --query-root data/market1501/test/query --gallery-root data/market1501/test/gallery
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcrtd.data import ImageFolderReID  # noqa: E402
from tcrtd.metrics import evaluate, k_reciprocal_rerank  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--query-root", required=True, type=Path)
    p.add_argument("--gallery-root", required=True, type=Path)
    p.add_argument("--k1", type=int, default=20)
    p.add_argument("--k2", type=int, default=6)
    p.add_argument("--lam", type=float, default=0.3)
    p.add_argument("--ignore-identity", nargs="*", default=["-1"])
    args = p.parse_args(argv)

    features = np.load(args.features)
    query, gallery = ImageFolderReID(args.query_root), ImageFolderReID(args.gallery_root)
    distance = k_reciprocal_rerank(torch.from_numpy(features["query"]), torch.from_numpy(features["gallery"]),
                                   args.k1, args.k2, args.lam)
    # Build the graph on the full gallery, then apply the evaluator junk rule.
    keep = np.array([i not in args.ignore_identity for i in gallery.identity_names()])
    pick = lambda values: [v for v, k in zip(values, keep) if k]
    ap, rank1 = evaluate(-distance[:, keep], query.identity_names(), pick(gallery.identity_names()),
                         pick(gallery.keys()), query.cameras, pick(gallery.cameras))
    print(json.dumps({"mAP": float(np.nanmean(ap)) * 100, "rank1": float(np.nanmean(rank1)) * 100}))


if __name__ == "__main__":
    main()
