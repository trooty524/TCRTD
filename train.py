"""Train M0, TCRTD, Smooth-AP, or Smooth-AP + TCRTD on one ReID training split.

Example (DINOv2-B/14, Market-1501, the main configuration of the paper):

    python train.py --train-root data/market1501/train --output runs/market_b14_tcrtd \
        --backbone dinov2_b14 --objective tcrtd --epochs 6 --seed 13
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tcrtd.data import ImageFolderReID, PKSampler, eval_transform, train_transform
from tcrtd.losses import hard_margin_triplet, smooth_ap, soft_margin_triplet
from tcrtd.model import BACKBONES, ReIDModel, is_resnet
from tcrtd.relation import RelationConfig, build_relation_target, ema_update, relation_losses


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-root", required=True, type=Path, help="root/<identity>/<image>")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--backbone", choices=sorted(BACKBONES), default="dinov2_b14")
    p.add_argument("--weights", default=None, help="local pretrained weights (downloaded if omitted)")
    p.add_argument("--objective", choices=("m0", "tcrtd", "smooth_ap", "tcrtd_smooth_ap"), default="tcrtd")
    p.add_argument("--epochs", type=int, required=True, help="training budget; the last checkpoint is kept")
    p.add_argument("--seed", type=int, default=13)

    r = p.add_argument_group("relation target (TCRTD)")
    r.add_argument("--relation", choices=("junk_aware", "legacy", "fixed_support"), default="junk_aware")
    r.add_argument("--rank-policy", choices=("student_score", "candidate_order"), default="student_score")
    r.add_argument("--topology-weight", type=float, default=0.5)
    r.add_argument("--swap-weight", type=float, default=0.3)
    r.add_argument("--candidates", type=int, default=64)
    r.add_argument("--reciprocal-k", type=int, default=10)
    r.add_argument("--raw-weight", type=float, default=0.5)
    r.add_argument("--jaccard-weight", type=float, default=0.5)
    r.add_argument("--teacher-temperature", type=float, default=0.1)
    r.add_argument("--student-temperature", type=float, default=0.1)
    r.add_argument("--ema", type=float, default=0.995)
    r.add_argument("--smooth-ap-temperature", type=float, default=0.01)
    r.add_argument("--smooth-ap-weight", type=float, default=1.0)

    o = p.add_argument_group("optimization (defaults follow the backbone family)")
    o.add_argument("--p", type=int, help="identities per batch (DINOv2 32, ResNet 16)")
    o.add_argument("--k", type=int, default=4, help="images per identity")
    o.add_argument("--sampler", choices=("camera_aware", "random"), default="camera_aware")
    o.add_argument("--schedule-epochs", type=int, default=80, help="length of the learning-rate schedule")
    o.add_argument("--optimizer", choices=("adamw", "adam"), help="DINOv2 adamw, ResNet adam")
    o.add_argument("--lr-schedule", choices=("cosine", "step"), help="DINOv2 cosine, ResNet step")
    o.add_argument("--lr", type=float, default=3.5e-4, help="ResNet learning rate")
    o.add_argument("--weight-decay", type=float, help="DINOv2 backbone 0.05, ResNet 5e-4")
    o.add_argument("--warmup-epochs", type=int, help="DINOv2 1, ResNet 10")
    o.add_argument("--workers", type=int, default=8)
    o.add_argument("--max-batches", type=int, help="limit batches per epoch (smoke tests)")
    args = p.parse_args(argv)

    resnet = is_resnet(args.backbone)
    defaults = dict(p=16 if resnet else 32, optimizer="adam" if resnet else "adamw",
                    lr_schedule="step" if resnet else "cosine",
                    weight_decay=5e-4 if resnet else 0.05, warmup_epochs=10 if resnet else 1)
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def set_seed(seed: int) -> torch.Generator:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(_: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_optimizer(model: ReIDModel, args) -> torch.optim.Optimizer:
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(("bnneck.", "classifier."))]
    head_ids = {id(p) for p in head}
    if not is_resnet(args.backbone):
        # DINOv2: backbone 1e-5, last four blocks 5e-5, head 3e-4.
        blocks = len(model.backbone.vit.blocks)
        last = [p for n, p in model.named_parameters()
                if p.requires_grad and n.startswith("backbone.vit.blocks.")
                and int(n.split(".")[3]) >= blocks - 4]
        last_ids = {id(p) for p in last}
        rest = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids | last_ids]
        groups = [dict(params=rest, lr=1e-5, weight_decay=args.weight_decay),
                  dict(params=last, lr=5e-5, weight_decay=args.weight_decay),
                  dict(params=head, lr=3e-4, weight_decay=5e-4)]
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if args.optimizer == "adam":
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return torch.optim.AdamW([dict(params=rest, lr=args.lr, weight_decay=args.weight_decay),
                              dict(params=head, lr=args.lr, weight_decay=5e-4)])


def lr_factor(step: int, batches: int, args) -> float:
    """Learning-rate multiplier at optimizer step ``step`` (``batches`` steps per epoch)."""
    warmup, horizon = args.warmup_epochs, args.schedule_epochs
    if not is_resnet(args.backbone):
        # DINOv2: per-step linear warmup, then cosine decay to 1% of the base rate.
        warm, total = warmup * batches, horizon * batches
        if step < warm:
            return min(1.0, (step + 1) / warm)
        t = min(1.0, (step - warm) / (total - warm))
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * t))
    progress = step / batches   # ResNet: schedule in fractional epochs
    if progress < warmup:
        return 0.1 + 0.9 * progress / warmup
    if args.lr_schedule == "step":
        return 1.0 if progress < 40 else (0.1 if progress < 70 else 0.01)
    t = min(1.0, (progress - warmup) / max(1, horizon - warmup))
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * t))


@torch.no_grad()
def encode(model: ReIDModel, dataset, device, batch_size: int = 128, workers: int = 8) -> torch.Tensor:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    out = []
    for images, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model.describe(images.to(device, non_blocking=True)).float())
    return torch.cat(out)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    generator = set_seed(args.seed)
    device = torch.device("cuda")
    resnet = is_resnet(args.backbone)
    image_size = BACKBONES[args.backbone][2]

    train_set = ImageFolderReID(args.train_root, train_transform(image_size, "bot" if resnet else "dinov2"))
    memory_set = ImageFolderReID(args.train_root, eval_transform(image_size))
    sampler = PKSampler(train_set, args.p, args.k, args.seed, camera_aware=args.sampler == "camera_aware")
    loader = DataLoader(train_set, batch_size=args.p * args.k, sampler=sampler, drop_last=True,
                        num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker,
                        generator=generator, persistent_workers=args.workers > 0)
    batches = len(loader) if args.max_batches is None else min(len(loader), args.max_batches)

    model = ReIDModel(args.backbone, train_set.num_classes, args.weights).to(device)
    optimizer = build_optimizer(model, args)
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    uses_relation = args.objective in ("tcrtd", "tcrtd_smooth_ap")
    uses_smooth_ap = args.objective in ("smooth_ap", "tcrtd_smooth_ap")
    teacher = copy.deepcopy(model).eval().requires_grad_(False) if uses_relation else None
    config = RelationConfig(args.candidates, args.reciprocal_k, args.raw_weight, args.jaccard_weight,
                            args.teacher_temperature, args.student_temperature, args.relation, args.rank_policy)
    labels = torch.tensor(train_set.labels, device=device)
    cameras = torch.tensor(train_set.cameras, device=device)

    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        target = None
        if uses_relation:   # refresh the full-training-set relation target with the EMA teacher
            memory = encode(teacher, memory_set, device, workers=args.workers)
            target = build_relation_target(memory, labels, cameras, config)
        sampler.set_epoch(epoch - 1)
        model.train()
        sums = dict(loss=0.0, ce=0.0, triplet=0.0, topology=0.0, swap=0.0, smooth_ap=0.0)
        for step, (images, y, indices) in enumerate(loader):
            if step >= batches:
                break
            factor = lr_factor((epoch - 1) * batches + step, batches, args)
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * factor
            images, y, indices = images.to(device, non_blocking=True), y.to(device), indices.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feature, descriptor, logits = model(images)
            ce = F.cross_entropy(logits.float(), y, label_smoothing=0.1)
            triplet = hard_margin_triplet(feature, y) if resnet else soft_margin_triplet(descriptor, y)
            loss = ce + triplet
            terms = dict(ce=ce, triplet=triplet)
            if target is not None:
                topology, swap, _ = relation_losses(descriptor, indices, target)
                loss = loss + args.topology_weight * topology + args.swap_weight * swap
                terms.update(topology=topology, swap=swap)
            if uses_smooth_ap:
                sap = smooth_ap(descriptor, y, args.smooth_ap_temperature)
                loss = loss + args.smooth_ap_weight * sap
                terms.update(smooth_ap=sap)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if teacher is not None:
                ema_update(teacher, model, args.ema)
            sums["loss"] += float(loss.detach())
            for name, value in terms.items():
                sums[name] += float(value.detach())
        row = {"epoch": epoch, "seconds": round(time.time() - started, 1),
               **{name: value / batches for name, value in sums.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)

    torch.save({"backbone": args.backbone, "num_classes": train_set.num_classes,
                "state_dict": model.state_dict(), "args": {k: str(v) for k, v in vars(args).items()}},
               args.output / "model.pth")
    (args.output / "history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
