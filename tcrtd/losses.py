"""Supervised ReID losses used by the baseline (M0) and the Smooth-AP comparator."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _hardest(distances: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    same = labels[:, None] == labels[None, :]
    positive = same.clone()
    positive.fill_diagonal_(False)
    hardest_positive = distances.masked_fill(~positive, -torch.inf).amax(dim=1)
    hardest_negative = distances.masked_fill(same, torch.inf).amin(dim=1)
    return hardest_positive, hardest_negative


def soft_margin_triplet(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Batch-hard soft-margin triplet on L2-normalized features (DINOv2 recipe)."""
    x = F.normalize(features.float(), dim=1)
    ap, an = _hardest(torch.cdist(x, x), labels)
    return F.softplus(ap - an).mean()


def hard_margin_triplet(features: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Batch-hard triplet with margin on raw pooled features (BoT recipe for ResNet)."""
    x = features.float()
    ap, an = _hardest(torch.cdist(x, x), labels)
    return F.relu(ap - an + margin).mean()


def smooth_ap(descriptors: torch.Tensor, labels: torch.Tensor, temperature: float = 0.01) -> torch.Tensor:
    """Smooth-AP (Brown et al., ECCV 2020): 1 - mean AP with a sigmoid relaxation of ranks."""
    x = F.normalize(descriptors, dim=1)
    scores = x @ x.T
    n = scores.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=scores.device)
    positive = (labels[:, None] == labels[None, :]) & ~eye
    pair_valid = (~eye)[:, :, None] & (~eye)[None, :, :]
    higher = torch.sigmoid((scores[:, :, None] - scores[:, None, :]) / temperature)
    rank = 1.0 + (higher * pair_valid).sum(dim=1)
    positive_rank = 1.0 + (higher * pair_valid * positive[:, :, None]).sum(dim=1)
    count = positive.sum(dim=1)
    ap = (positive_rank / rank * positive).sum(dim=1) / count.clamp_min(1)
    has_positive = count > 0
    if not bool(has_positive.any()):
        return descriptors.sum() * 0.0
    return 1.0 - ap[has_positive].mean()
