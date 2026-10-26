"""Materialize a local animal ReID split manifest.

The paper's sealed LeopardID2022 and WhaleSharkID test manifests are not
distributed in this repository. An authorized user can keep such a manifest
locally and use this tool to create the image-folder layout consumed by
``train.py`` and ``evaluate.py``::

    python tools/materialize_animal_split.py \
      --manifest /private/LeopardID2022_manifest.json \
      --source-root /data/LeopardID2022 \
      --output data/animals/leopardid2022

Manifest paths are relative to ``--source-root`` and each entry names its
identity explicitly. The tool copies files, writes the manifest used and an
audit summary, and refuses absolute paths or path traversal.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

SPLITS = ("train", "query", "gallery")


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    for key in ("dataset", "version", *SPLITS):
        if key not in data:
            raise ValueError(f"manifest is missing '{key}'")
    for split in SPLITS:
        if not isinstance(data[split], list):
            raise ValueError(f"manifest field '{split}' must be a list")
        for index, entry in enumerate(data[split]):
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError(f"{split}[{index}] must contain a string path")
            if not isinstance(entry.get("identity"), str) or not entry["identity"]:
                raise ValueError(f"{split}[{index}] must contain a non-empty identity")
    return data


def _safe_source(source_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"manifest paths must be relative paths without '..': {relative}")
    root = source_root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"manifest path escapes source root: {relative}")
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest image does not exist: {resolved}")
    return resolved


def _copy_entry(source: Path, output: Path, identity: str) -> None:
    if not identity or Path(identity).name != identity or identity in {".", ".."}:
        raise ValueError(f"identity must be a single safe directory name: {identity!r}")
    destination = output / identity / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"duplicate destination filename for identity: {destination}")
    shutil.copy2(source, destination)


def materialize_manifest(manifest_path: Path, source_root: Path, output: Path) -> dict[str, Any]:
    """Copy the three manifest splits and return the written audit summary."""

    manifest = _load_manifest(manifest_path)
    source_root = source_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    identities: dict[str, int] = {}
    records: dict[str, list[dict[str, str]]] = {}
    for split in SPLITS:
        split_root = output / "train" if split == "train" else output / "test" / split
        records[split] = []
        seen: set[tuple[str, str]] = set()
        for entry in manifest[split]:
            source = _safe_source(source_root, entry["path"])
            identity = entry["identity"]
            key = (identity, source.name)
            if key in seen:
                raise ValueError(f"duplicate manifest entry: {split}/{identity}/{source.name}")
            seen.add(key)
            _copy_entry(source, split_root, identity)
            records[split].append({"path": entry["path"], "identity": identity})
        counts[split] = len(records[split])
        identities[split] = len({item["identity"] for item in records[split]})

    audit = {
        "dataset": manifest["dataset"],
        "version": manifest["version"],
        "source_root": str(source_root),
        "counts": counts,
        "identities": identities,
        "files": records,
    }
    (output / "manifest_used.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(materialize_manifest(args.manifest, args.source_root, args.output), indent=2))


if __name__ == "__main__":
    main()
