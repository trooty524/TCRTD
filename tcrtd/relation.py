"""Relation targets and losses used by TCRTD.

The EMA teacher encodes the training set. Reciprocal neighborhoods, fused
cosine/Jaccard distances and the AP-swap loss define the target seen by the
student. The three target policies are ``legacy``, ``junk_aware`` and
``fixed_support``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

RelationPolicy = Literal["legacy", "junk_aware", "fixed_support"]
RankPolicy = Literal["student_score", "candidate_order"]


@dataclass(frozen=True)
class RelationConfig:
    candidates: int = 64            # K
    reciprocal_k: int = 10          # k of the reciprocal sets
    raw_weight: float = 0.5         # weight of (1 - cosine) in the fused distance
    jaccard_weight: float = 0.5     # weight of the Jaccard distance
    teacher_temperature: float = 0.1
    student_temperature: float = 0.1
    policy: RelationPolicy = "junk_aware"
    rank_policy: RankPolicy = "student_score"
    block_size: int = 2048          # rows per block when building the k-NN graph


class RelationTarget:
    """Teacher memory and reciprocal graph for one epoch."""

    def __init__(self, memory, labels, cameras, reciprocal, config: RelationConfig):
        self.memory = memory            # [N, d], L2-normalized teacher descriptors
        self.labels = labels            # [N] identity labels
        self.cameras = cameras          # [N] camera ids, or None
        self.reciprocal = reciprocal    # [N, k] reciprocal neighbors, -1 = empty slot
        self.config = config

    def _junk(self, anchors: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
        """Same identity and same camera as the anchor (ignored by the evaluator)."""
        same_id = self.labels[anchors].unsqueeze(1) == self.labels[others]
        same_cam = self.cameras[anchors].unsqueeze(1) == self.cameras[others]
        return same_id & same_cam

    @torch.no_grad()
    def targets(self, anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-K candidates of each anchor and the teacher distribution p over them."""
        cfg = self.config
        similarity = self.memory[anchors] @ self.memory.T
        similarity.scatter_(1, anchors[:, None], -torch.inf)
        if cfg.policy == "junk_aware":
            everyone = torch.arange(self.memory.shape[0], device=anchors.device)
            similarity.masked_fill_(self._junk(anchors, everyone), -torch.inf)
        cand_sim, candidates = similarity.topk(cfg.candidates, dim=1, largest=True, sorted=True)

        # Jaccard distance of the reciprocal sets.
        anchor_sets = self.reciprocal[anchors][:, None, :]
        cand_sets = self.reciprocal[candidates]
        anchor_valid, cand_valid = anchor_sets >= 0, cand_sets >= 0
        shared = (anchor_sets[..., :, None] == cand_sets[..., None, :]) \
            & anchor_valid[..., :, None] & cand_valid[..., None, :]
        intersection = shared.any(dim=-1).sum(dim=-1, dtype=torch.float32)
        union = anchor_valid.sum(-1, dtype=torch.float32) + cand_valid.sum(-1, dtype=torch.float32) - intersection
        jaccard = torch.where(union > 0, 1.0 - intersection / union, torch.ones_like(union))

        # Fused teacher distance and relation target.
        distance = cfg.raw_weight * (1.0 - cand_sim) + cfg.jaccard_weight * jaccard
        p = torch.softmax(-distance / cfg.teacher_temperature, dim=1)

        if cfg.policy == "fixed_support":
            # Evaluator-compatible projection on the legacy support.
            retained = p.masked_fill(self._junk(anchors, candidates), 0.0)
            mass = retained.sum(dim=1, keepdim=True)
            p = torch.where(mass > 0, retained / mass.clamp_min(torch.finfo(p.dtype).tiny), p)
        return candidates, p


@torch.no_grad()
def build_relation_target(descriptors, labels, cameras, config: RelationConfig) -> RelationTarget:
    """Build the k-reciprocal graph over the full training memory (blockwise, no N x N matrix)."""
    memory = F.normalize(descriptors.float(), dim=1)
    count = memory.shape[0]
    if config.policy != "legacy" and (cameras is None or bool((cameras < 0).any())):
        raise ValueError(f"relation policy '{config.policy}' needs a camera label for every image")
    exclude_junk = config.policy == "junk_aware"

    neighbors = []
    for start in range(0, count, config.block_size):
        rows = torch.arange(start, min(count, start + config.block_size), device=memory.device)
        similarity = memory[rows] @ memory.T
        similarity[torch.arange(len(rows)), rows] = -torch.inf
        if exclude_junk:
            junk = (labels[rows, None] == labels[None, :]) & (cameras[rows, None] == cameras[None, :])
            similarity.masked_fill_(junk, -torch.inf)
        neighbors.append(similarity.topk(config.reciprocal_k, dim=1).indices)
    neighbors = torch.cat(neighbors)

    # j is a reciprocal neighbor of i if j in N_k(i) and i in N_k(j).
    mutual = (neighbors[neighbors] == torch.arange(count, device=memory.device)[:, None, None]).any(dim=-1)
    reciprocal = torch.where(mutual, neighbors, torch.full_like(neighbors, -1))
    return RelationTarget(memory, labels, cameras, reciprocal, config)


def relation_losses(student_descriptors, anchors, target: RelationTarget):
    """Topology KL, AP-swap loss, and the share of anchors whose list holds a positive."""
    cfg = target.config
    candidates, p = target.targets(anchors)
    student = F.normalize(student_descriptors.float(), dim=1)
    scores = torch.einsum("bd,bkd->bk", student, target.memory[candidates])
    log_q = F.log_softmax(scores / cfg.student_temperature, dim=1)
    topology = F.kl_div(log_q, p, reduction="batchmean")

    relevant = target.labels[candidates] == target.labels[anchors][:, None]
    valid = None
    if cfg.policy == "junk_aware":
        junk = target._junk(anchors, candidates)
        relevant, valid = relevant & ~junk, ~junk
    swap = ap_swap_loss(scores, relevant, valid, rank_policy=cfg.rank_policy)
    coverage = relevant.any(dim=1).float().mean()
    return topology, swap, coverage


def ap_swap_loss(scores, relevant, valid=None, *, rank_policy: RankPolicy = "student_score"):
    """Logistic pair loss weighted by the exact AP change of swapping each positive-negative pair.

    For a list with relevance y_r, prefix counts C_r and P positives, moving a
    positive at rank a down to rank b > a changes AP by
        (1/P) [C_b / b - C_a / a - sum_{a<r<b} y_r / r],
    and moving it up to rank b < a changes AP by
        (1/P) [(C_b + 1) / b - C_a / a + sum_{b<r<a} y_r / r].
    """
    if rank_policy == "student_score":
        order = torch.argsort(scores.detach(), dim=1, descending=True, stable=True)
        scores, relevant = scores.gather(1, order), relevant.gather(1, order)
        valid = None if valid is None else valid.gather(1, order)
    weighted = scores.sum() * 0.0
    total = scores.sum() * 0.0
    ranks = torch.arange(1, scores.shape[1] + 1, device=scores.device, dtype=scores.dtype)
    for row in range(scores.shape[0]):
        keep = torch.ones_like(relevant[row]) if valid is None else valid[row]
        s, y = scores[row, keep], relevant[row, keep]
        r = ranks[: s.numel()]
        pos, neg = torch.where(y)[0], torch.where(~y)[0]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        yf = y.to(scores.dtype)
        prefix = yf.cumsum(0)                  # C_r
        inv_prefix = (yf / r).cumsum(0)        # sum_{t<=r} y_t / t
        a, b = pos[:, None], neg[None, :]
        down = (prefix[b] / r[b] - prefix[a] / r[a]
                - (inv_prefix[b - 1] - inv_prefix[a]))
        up = ((prefix[b] + 1.0) / r[b] - prefix[a] / r[a]
              + (inv_prefix[a - 1] - inv_prefix[b]))
        weight = torch.where(a < b, down, up).abs() / pos.numel()
        margin = s[pos][:, None] - s[neg][None, :]
        weighted = weighted + (weight * F.softplus(-margin)).sum()
        total = total + weight.sum()
    return weighted / total.clamp_min(torch.finfo(scores.dtype).eps)


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """theta_bar <- m * theta_bar + (1 - m) * theta; buffers are copied."""
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(momentum).add_(s, alpha=1.0 - momentum)
    for t, s in zip(teacher.buffers(), student.buffers()):
        t.copy_(s)


@torch.no_grad()
def target_masses(target: RelationTarget, block: int = 256) -> dict[str, float]:
    """Mean relevant (mu), junk (rho), and negative (nu) mass of the relation target."""
    n = target.memory.shape[0]
    mu = rho = 0.0
    for start in range(0, n, block):
        anchors = torch.arange(start, min(n, start + block), device=target.memory.device)
        candidates, p = target.targets(anchors)
        same_id = target.labels[candidates] == target.labels[anchors][:, None]
        same_cam = target.cameras[candidates] == target.cameras[anchors][:, None]
        mu += float((p * (same_id & ~same_cam)).sum())
        rho += float((p * (same_id & same_cam)).sum())
    return {"mu": mu / n, "rho": rho / n, "nu": 1.0 - (mu + rho) / n}
