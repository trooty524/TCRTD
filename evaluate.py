"""Evaluate a checkpoint with plain cosine matching.

The command writes ``metrics.json`` and ``per_query.npz``. Market-1501 junk
images (identity -1) are ignored by default. ``--identity-only`` removes the
camera rule for animal data; ``--all-vs-all`` uses one folder as both query and
gallery and skips each query itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tcrtd.data import ImageFolderReID, eval_transform
from tcrtd.metrics import evaluate
from tcrtd.model import ReIDModel
from train import encode


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--query-root", required=True, type=Path)
    p.add_argument("--gallery-root", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--identity-only", action="store_true", help="no camera labels: every same-identity image is relevant")
    p.add_argument("--all-vs-all", action="store_true", help="query folder is also the gallery; self-matches excluded")
    p.add_argument("--ignore-identity", nargs="*", default=["-1"],
                   help="gallery identities ignored by the metric (default: -1, the Market-1501 junk "
                        "images); pass the flag without values to keep every gallery image")
    p.add_argument("--save-features", action="store_true", help="also save descriptors (for tools/rerank.py)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    if args.gallery_root is None and not args.all_vs_all:
        p.error("--gallery-root is required unless --all-vs-all is set")

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ReIDModel(checkpoint["backbone"], checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    transform = eval_transform(model.image_size)
    query = ImageFolderReID(args.query_root, transform)
    gallery = query if args.all_vs_all else ImageFolderReID(args.gallery_root, transform)
    zq = encode(model, query, device, workers=args.workers)
    zg = zq if args.all_vs_all else encode(model, gallery, device, workers=args.workers)
    similarity = (zq @ zg.T).cpu().numpy()
    gallery_ids, gallery_keys, gallery_cams = gallery.identity_names(), gallery.keys(), gallery.cameras
    if args.ignore_identity:
        keep = np.array([i not in args.ignore_identity for i in gallery_ids])
        similarity = similarity[:, keep]
        gallery_ids, gallery_keys, gallery_cams = ([v for v, k in zip(values, keep) if k]
                                                   for values in (gallery_ids, gallery_keys, gallery_cams))

    if args.all_vs_all:
        ap, rank1 = _evaluate_without_self(similarity, query)
    else:
        cameras = not args.identity_only
        ap, rank1 = evaluate(similarity, query.identity_names(), gallery_ids, gallery_keys,
                             query.cameras if cameras else None, gallery_cams if cameras else None)

    metrics = {"mAP": float(np.nanmean(ap)) * 100, "rank1": float(np.nanmean(rank1)) * 100,
               "queries": int(np.isfinite(ap).sum()), "gallery": similarity.shape[1]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez_compressed(args.output / "per_query.npz", keys=np.asarray(query.keys()),
                        identities=np.asarray(query.identity_names()), ap=ap, rank1=rank1)
    if args.save_features:
        np.savez_compressed(args.output / "features.npz", query=zq.cpu().numpy(), gallery=zg.cpu().numpy())
    print(json.dumps(metrics))


def _evaluate_without_self(similarity: np.ndarray, dataset: ImageFolderReID):
    """Rank the other images in the same set for each query."""
    ids = np.asarray(dataset.identity_names())
    keys = dataset.keys()
    ap = np.full(len(ids), np.nan)
    rank1 = np.full(len(ids), np.nan)
    for q in range(len(ids)):
        others = np.arange(len(ids)) != q
        sub_ap, sub_r1 = evaluate(similarity[q:q + 1, others], ids[q:q + 1], ids[others],
                                  [k for k, keep in zip(keys, others) if keep], workers=1)
        ap[q], rank1[q] = sub_ap[0], sub_r1[0]
    return ap, rank1


if __name__ == "__main__":
    main()
