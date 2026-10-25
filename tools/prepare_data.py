"""Arrange an official ReID release as root/<identity>/<image> and audit train/test content overlap.

    python tools/prepare_data.py --dataset market1501 --source /data/Market-1501-v15.09.15 --output data/market1501
    python tools/prepare_data.py --dataset msmt17     --source /data/MSMT17_V1           --output data/msmt17
    python tools/prepare_data.py --dataset nightreid  --source /data/NightReID           --output data/nightreid \
        --exclude-identities protocols/nightreid_excluded_identities.txt

Output: <output>/train, <output>/test/query, <output>/test/gallery, and
audit.json. A training image is removed when its bytes or its decoded RGB
pixels equal those of any test image; identities listed in
--exclude-identities are removed from training entirely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

LAYOUTS = {  # dataset: (train, query, gallery) folders of the official release
    "market1501": ("bounding_box_train", "query", "bounding_box_test"),
    "nightreid": ("bounding_box_train", "query", "bounding_box_test"),
}


def identity_from_name(name: str) -> str:
    """Leading person id of Market-1501 ('0002_c1s1_...', '-1_c1...') and NightReID ('0001L1C001') files."""
    match = re.match(r"^(-?\d+)", name)
    if match is None:
        raise ValueError(f"cannot parse an identity from {name}")
    return match.group(1)


def flat_split(folder: Path) -> list[tuple[Path, str]]:
    return [(p, identity_from_name(p.name)) for p in sorted(folder.iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")]


def msmt17_split(source: Path, lists: list[str], images: str) -> list[tuple[Path, str]]:
    rows = []
    for name in lists:
        for line in (source / name).read_text().splitlines():
            if line.strip():
                relative = line.split()[0]
                rows.append((source / images / relative, relative.split("/")[0]))
    return rows


def digests(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    return hashlib.sha256(raw).hexdigest(), hashlib.sha256(pixels.tobytes() + str(pixels.shape).encode()).hexdigest()


def place(rows: list[tuple[Path, str]], root: Path) -> None:
    for source, identity in rows:
        target = root / identity / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("market1501", "msmt17", "nightreid"), required=True)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--exclude-identities", type=Path, help="text file, one training identity per line")
    args = p.parse_args(argv)

    if args.dataset == "msmt17":
        train = msmt17_split(args.source, ["list_train.txt", "list_val.txt"], "train")
        query = msmt17_split(args.source, ["list_query.txt"], "test")
        gallery = msmt17_split(args.source, ["list_gallery.txt"], "test")
    else:
        train, query, gallery = (flat_split(args.source / folder) for folder in LAYOUTS[args.dataset])

    excluded = set()
    if args.exclude_identities is not None:
        excluded = {line.strip() for line in args.exclude_identities.read_text().splitlines() if line.strip()}

    test_hashes: set[str] = set()
    for path, _ in query + gallery:
        test_hashes.update(digests(path))
    kept, duplicates = [], []
    for path, identity in train:
        if identity in excluded:
            continue
        if set(digests(path)) & test_hashes:
            duplicates.append(f"{identity}/{path.name}")
        else:
            kept.append((path, identity))

    place(kept, args.output / "train")
    place(query, args.output / "test" / "query")
    place(gallery, args.output / "test" / "gallery")
    audit = {
        "dataset": args.dataset,
        "train": {"images": len(kept), "identities": len({i for _, i in kept})},
        "query": len(query), "gallery": len(gallery),
        "excluded_identities": sorted(excluded),
        "removed_duplicates_of_test_images": duplicates,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps({k: v for k, v in audit.items() if k != "removed_duplicates_of_test_images"}
                     | {"removed_duplicates": len(duplicates)}, indent=1))


if __name__ == "__main__":
    main()
