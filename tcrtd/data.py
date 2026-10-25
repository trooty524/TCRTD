"""Image-folder datasets, camera labels, PK sampling, and transforms."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NIGHTREID_CAMERAS = {"L1": 0, "L2": 1, "L3": 2, "R1": 3, "R2": 4, "R3": 5}


def parse_camera(filename: str) -> int:
    """Camera id from a Market-1501, MSMT17, or NightReID file name; -1 if unknown."""
    name = Path(filename).name
    if m := re.search(r"_c(\d+)[_s]", name, flags=re.IGNORECASE):          # Market-1501
        return int(m.group(1))
    if m := re.search(r"\d{4}([LR][123])C\d+", name, flags=re.IGNORECASE):  # NightReID
        return NIGHTREID_CAMERAS[m.group(1).upper()]
    if m := re.match(r"^\d+_\d+_(\d{2})_", name):                           # MSMT17
        return int(m.group(1))
    return -1


class ImageFolderReID(Dataset):
    """A ReID split stored as ``root/<identity>/<image>``.

    Labels follow the sorted identity folders and images are sorted within each
    folder, so indices are stable across runs; ``__getitem__`` also returns the
    index, which addresses the teacher memory.
    """

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.identities = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.paths: list[Path] = []
        self.labels: list[int] = []
        self.cameras: list[int] = []
        for label, identity in enumerate(self.identities):
            for path in sorted((self.root / identity).iterdir(), key=lambda p: p.name):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.paths.append(path)
                    self.labels.append(label)
                    self.cameras.append(parse_camera(path.name))

    @property
    def num_classes(self) -> int:
        return len(self.identities)

    def keys(self) -> list[str]:
        """Relative paths; used as a deterministic tie-breaker when ranking."""
        return [p.relative_to(self.root).as_posix() for p in self.paths]

    def identity_names(self) -> list[str]:
        return [self.identities[label] for label in self.labels]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index], index


class PKSampler(Sampler[int]):
    """P identities x K images per batch.

    With ``camera_aware=True`` the K images of an identity are first drawn from
    distinct cameras, so every anchor sees a cross-camera positive.
    """

    def __init__(self, dataset: ImageFolderReID, p: int, k: int, seed: int, camera_aware: bool):
        self.dataset, self.p, self.k, self.seed = dataset, p, k, int(seed)
        self.camera_aware, self.epoch = camera_aware, 0
        self.by_identity: dict[int, list[int]] = {}
        self.by_camera: dict[int, dict[int, list[int]]] = {}
        for index, (label, camera) in enumerate(zip(dataset.labels, dataset.cameras)):
            self.by_identity.setdefault(label, []).append(index)
            self.by_camera.setdefault(label, {}).setdefault(camera, []).append(index)
        self.identities = [
            label for label, indices in self.by_identity.items()
            if len(indices) >= k and (not camera_aware or self._num_cameras(label) >= 2)
        ]
        if len(self.identities) < p:
            self.identities = list(self.by_identity)

    def _num_cameras(self, label: int) -> int:
        return sum(1 for camera, idx in self.by_camera[label].items() if camera >= 0 and idx)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _draw(self, label: int, rng: np.random.RandomState) -> list[int]:
        indices = self.by_identity[label]
        if not self.camera_aware:
            return rng.choice(indices, size=self.k, replace=len(indices) < self.k).tolist()
        groups = {c: idx for c, idx in self.by_camera[label].items() if c >= 0 and idx}
        cameras = rng.choice(sorted(groups), size=min(self.k, len(groups)), replace=False)
        chosen = [int(rng.choice(groups[int(c)])) for c in cameras]
        if len(chosen) < self.k:
            unused = [i for i in indices if i not in set(chosen)] or indices
            extra = rng.choice(unused, size=self.k - len(chosen), replace=len(unused) < self.k - len(chosen))
            chosen.extend(int(i) for i in extra)
        return chosen

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        order: list[int] = []
        for _ in range(len(self.dataset) // (self.p * self.k)):
            for label in rng.choice(self.identities, size=min(self.p, len(self.identities)), replace=False):
                order.extend(self._draw(int(label), rng))
        return iter(order)

    def __len__(self) -> int:
        return (len(self.dataset) // (self.p * self.k)) * self.p * self.k


def train_transform(image_size: tuple[int, int], recipe: str) -> transforms.Compose:
    """``dinov2``: flip, light color jitter, random erasing; ``bot``: flip, pad-crop, erasing (BoT)."""
    if recipe == "bot":
        return transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0),
        ])
    return transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
    ])


def eval_transform(image_size: tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
