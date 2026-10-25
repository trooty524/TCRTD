"""Retrieval metrics, fixed k-reciprocal re-ranking, and identity-block bootstrap."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.nn import functional as F


def rank_galleries(similarity: np.ndarray, gallery_keys: list[str]) -> np.ndarray:
    """Descending order of each row; ties are broken by the gallery key."""
    key_order = np.argsort(np.asarray(gallery_keys), kind="stable")
    keyed = similarity.astype(np.float32)[:, key_order]
    return key_order[np.argsort(-keyed, axis=1, kind="stable")]


def evaluate(similarity, query_ids, gallery_ids, gallery_keys, query_cams=None, gallery_cams=None,
             block: int = 64, workers: int = 8):
    """Per-query AP and Rank-1.

    With cameras, gallery images of the query identity taken by the query camera
    are ignored (standard ReID junk rule); without cameras every same-identity
    image is relevant. Queries without a relevant gallery image get NaN.
    """
    similarity = np.asarray(similarity)
    query_ids, gallery_ids = np.asarray(query_ids), np.asarray(gallery_ids)
    use_cams = query_cams is not None
    if use_cams:
        query_cams, gallery_cams = np.asarray(query_cams), np.asarray(gallery_cams)
    ap = np.full(len(query_ids), np.nan)
    rank1 = np.full(len(query_ids), np.nan)

    def run(start: int) -> None:
        orders = rank_galleries(similarity[start:start + block], gallery_keys)
        for offset, order in enumerate(orders):
            q = start + offset
            same_id = gallery_ids == query_ids[q]
            if use_cams:
                same_cam = gallery_cams == query_cams[q]
                relevant, kept = same_id & ~same_cam, ~(same_id & same_cam)
            else:
                relevant, kept = same_id, np.ones_like(same_id)
            n_relevant = int(relevant.sum())
            if n_relevant == 0:
                continue
            hits = relevant[order[kept[order]]]
            hit_ranks = np.flatnonzero(hits) + 1
            ap[q] = float(np.mean(np.arange(1, n_relevant + 1) / hit_ranks))
            rank1[q] = float(hits[0])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, range(0, len(query_ids), block)))
    return ap, rank1


def identity_block_interval(delta, identities, reps: int = 10_000, seed: int = 20261010):
    """95% interval of a mean paired difference, resampling query identities with all their queries."""
    delta, identities = np.asarray(delta, dtype=np.float64), np.asarray(identities)
    units = np.unique(identities)
    sums = np.array([delta[identities == u].sum() for u in units])
    counts = np.array([(identities == u).sum() for u in units])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps // 250):
        draw = rng.integers(len(units), size=(250, len(units)))
        samples.append(sums[draw].sum(1) / counts[draw].sum(1))
    samples = np.concatenate(samples)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def k_reciprocal_rerank(query: torch.Tensor, gallery: torch.Tensor, k1: int = 20, k2: int = 6,
                        lam: float = 0.3, block: int = 64) -> np.ndarray:
    """k-reciprocal re-ranking (Zhong et al., CVPR 2017) with sparse encodings.

    Returns the re-ranked distance [num_query, num_gallery]; lower is better.
    Only the top-(k1+1) lists are materialized, so large galleries such as
    MSMT17 fit in memory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = torch.cat([query, gallery]).float()
    x = F.normalize(features.to(device), dim=1)
    n, nq, k = x.shape[0], query.shape[0], k1 + 1
    ranks = torch.empty((n, k), dtype=torch.int64)
    row_max = torch.empty(n)
    for s in range(0, n, block):
        d = torch.clamp(2.0 - 2.0 * (x[s:s + block] @ x.T), min=0.0)
        m = d.amax(dim=1).clamp_min(1e-12)
        ranks[s:s + block] = torch.argsort(d / m[:, None], dim=1, stable=True)[:, :k].cpu()
        row_max[s:s + block] = m.cpu()
    ranks, row_max = ranks.numpy(), row_max.numpy()
    xn = F.normalize(features, dim=1).numpy()

    def reciprocal(i: int, kk: int) -> list[int]:
        forward = ranks[i, :kk + 1]
        return forward[np.any(ranks[forward, :kk + 1] == i, axis=1)].tolist()

    encodings: list[dict[int, float]] = []
    for i in range(n):
        rec = reciprocal(i, k1)
        expanded = set(rec)
        for c in rec:
            half = set(reciprocal(c, max(1, k1 // 2)))
            if half and len(set(rec) & half) > 2.0 / 3.0 * len(half):
                expanded |= half
        cols = np.fromiter(sorted(expanded), dtype=np.int64)
        dist = np.clip(2.0 - 2.0 * (xn[cols] @ xn[i]), 0.0, None) / max(row_max[i], 1e-12)
        w = np.exp(-dist)
        encodings.append(dict(zip(cols.tolist(), (w / max(w.sum(), 1e-12)).tolist())))
    if k2 > 1:  # local query expansion
        expanded_enc = []
        for i in range(n):
            acc: dict[int, float] = {}
            for j in ranks[i, :k2]:
                for c, w in encodings[int(j)].items():
                    acc[c] = acc.get(c, 0.0) + w / k2
            expanded_enc.append(acc)
        encodings = expanded_enc

    inverse: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    totals = np.zeros(n)
    for r, enc in enumerate(encodings):
        for c, w in enc.items():
            inverse[c].append((r, w))
            totals[r] += w
    xq, xg = xn[:nq], xn[nq:]
    out = np.empty((nq, n - nq), dtype=np.float32)
    for i in range(nq):
        inter = np.zeros(n - nq)
        for c, w in encodings[i].items():
            for r, rw in inverse[c]:
                if r >= nq:
                    inter[r - nq] += min(w, rw)
        jaccard = 1.0 - inter / np.maximum(totals[i] + totals[nq:] - inter, 1e-12)
        original = np.clip(2.0 - 2.0 * (xq[i] @ xg.T), 0.0, None) / max(row_max[i], 1e-12)
        out[i] = (1.0 - lam) * jaccard + lam * original
    return out
