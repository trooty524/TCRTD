"""Checks of the exactness claims and of Theorem 1 / Corollary 1 of the paper.  Run: python -m pytest tests"""

import itertools
import math

import torch
from torch.nn import functional as F

from tcrtd.metrics import evaluate
from tcrtd.relation import RelationConfig, ap_swap_loss, build_relation_target, relation_losses


def list_ap(y: list[int]) -> float:
    hits, total = 0, 0.0
    for rank, rel in enumerate(y, start=1):
        if rel:
            hits += 1
            total += hits / rank
    return total / max(1, sum(y))


def test_ap_swap_weights_are_exact_ap_changes():
    """Every pair weight equals |AP(swapped list) - AP(list)|, normalized by the total weight."""
    torch.manual_seed(0)
    y = torch.tensor([1, 0, 0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.bool)
    scores = torch.linspace(1.0, 0.0, len(y))
    loss = ap_swap_loss(scores[None], y[None], rank_policy="candidate_order")
    expected_weights, margins = {}, {}
    base = list_ap(y.int().tolist())
    for p, n in itertools.product(torch.where(y)[0].tolist(), torch.where(~y)[0].tolist()):
        swapped = y.int().tolist()
        swapped[p], swapped[n] = swapped[n], swapped[p]
        expected_weights[(p, n)] = abs(list_ap(swapped) - base)
        margins[(p, n)] = float(scores[p] - scores[n])
    total = sum(expected_weights.values())
    expected = sum(w * math.log1p(math.exp(-margins[k])) for k, w in expected_weights.items()) / total
    assert abs(loss.item() - expected) < 1e-6


def test_projection_changes_target_by_twice_the_junk_mass():
    """Theorem 1(i): ||p^{not J} - p||_1 = 2 rho for the fixed-support projection."""
    torch.manual_seed(1)
    n = 400
    labels = torch.randint(0, 40, (n,))
    cameras = torch.randint(0, 3, (n,))
    x = torch.randn(n, 16) + 3 * torch.randn(40, 16)[labels]
    legacy = build_relation_target(x, labels, cameras, RelationConfig(policy="legacy"))
    fixed = build_relation_target(x, labels, cameras, RelationConfig(policy="fixed_support"))
    anchors = torch.arange(64)
    cand_l, p = legacy.targets(anchors)
    cand_f, p_proj = fixed.targets(anchors)
    assert torch.equal(cand_l, cand_f)
    rho = (p * legacy._junk(anchors, cand_l)).sum(1)
    keep = rho < 1
    assert torch.allclose((p_proj - p).abs().sum(1)[keep], 2 * rho[keep], atol=1e-5)

    # Corollary 1: relevant mass becomes mu / (1 - rho).
    relevant = (labels[cand_l] == labels[anchors][:, None]) & (cameras[cand_l] != cameras[anchors][:, None])
    mu, mu_proj = (p * relevant).sum(1), (p_proj * relevant).sum(1)
    assert torch.allclose(mu_proj[keep], (mu / (1 - rho))[keep], atol=1e-5)


def test_projection_redirects_descriptor_force():
    """Theorem 1(ii): -grad L(p_proj) = -grad L(p) + (rho / tau)(m_R - m_J)."""
    torch.manual_seed(2)
    k, d, tau = 12, 8, 0.1
    codebook = F.normalize(torch.randn(k, d), dim=1)
    junk = torch.zeros(k, dtype=torch.bool)
    junk[[1, 4, 7]] = True
    p = torch.softmax(torch.randn(k), 0)
    rho = p[junk].sum()
    p_proj = torch.where(junk, torch.zeros_like(p), p / (1 - rho))
    z = F.normalize(torch.randn(d), dim=0)

    def grad(target):
        zz = z.clone().requires_grad_(True)
        loss = F.kl_div(F.log_softmax(codebook @ zz / tau, 0), target, reduction="sum")
        loss.backward()
        return zz.grad

    m_r = (p_proj[:, None] * codebook).sum(0)
    m_j = ((p * junk / rho)[:, None] * codebook).sum(0)
    assert torch.allclose(-grad(p_proj), -grad(p) + rho / tau * (m_r - m_j), atol=1e-5)


def test_junk_aware_target_excludes_same_camera_frames():
    torch.manual_seed(3)
    n = 300
    labels = torch.randint(0, 30, (n,))
    cameras = torch.randint(0, 3, (n,))
    x = torch.randn(n, 16) + 3 * torch.randn(30, 16)[labels]
    target = build_relation_target(x, labels, cameras, RelationConfig(policy="junk_aware"))
    anchors = torch.arange(50)
    candidates, p = target.targets(anchors)
    assert not target._junk(anchors, candidates).any()
    assert torch.allclose(p.sum(1), torch.ones(50))
    topology, swap, coverage = relation_losses(torch.randn(50, 16), anchors, target)
    assert torch.isfinite(topology) and torch.isfinite(swap) and 0 <= coverage <= 1


def test_metric_ignores_same_camera_matches():
    # one query (id a, cam 0); gallery: same id same cam (junk), distractor, same id other cam
    sim = torch.tensor([[0.9, 0.8, 0.7]]).numpy()
    ap, rank1 = evaluate(sim, ["a"], ["a", "b", "a"], ["g0", "g1", "g2"], [0], [0, 1, 1])
    assert abs(ap[0] - 0.5) < 1e-9 and rank1[0] == 0.0
