import json
from pathlib import Path

import pytest

from tools.materialize_animal_split import materialize_manifest


def test_materialize_manifest_copies_split_and_writes_audit(tmp_path: Path):
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    for name in ("train.jpg", "query.jpg", "gallery.jpg"):
        (source / "images" / name).write_bytes(name.encode())

    manifest = {
        "dataset": "ExampleAnimal",
        "version": "test-v1",
        "train": [{"path": "images/train.jpg", "identity": "id-train"}],
        "query": [{"path": "images/query.jpg", "identity": "id-query"}],
        "gallery": [{"path": "images/gallery.jpg", "identity": "id-gallery"}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "prepared"

    audit = materialize_manifest(manifest_path, source, output)

    assert (output / "train" / "id-train" / "train.jpg").read_bytes() == b"train.jpg"
    assert (output / "test" / "query" / "id-query" / "query.jpg").exists()
    assert (output / "test" / "gallery" / "id-gallery" / "gallery.jpg").exists()
    assert audit["counts"] == {"train": 1, "query": 1, "gallery": 1}
    assert json.loads((output / "audit.json").read_text(encoding="utf-8"))["dataset"] == "ExampleAnimal"


def test_materialize_manifest_rejects_path_traversal(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "dataset": "ExampleAnimal",
            "version": "test-v1",
            "train": [{"path": "../outside.jpg", "identity": "id"}],
            "query": [],
            "gallery": [],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative path"):
        materialize_manifest(manifest_path, tmp_path / "source", tmp_path / "prepared")
