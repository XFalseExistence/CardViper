"""Read-only, deterministic audit of a local Joshua v2 YOLO export."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile

from PIL import Image
import yaml

from .labels import LABELS

SOURCE_ID = "playing-cards-seed"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _names(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and set(raw) == set(range(len(raw))):
        return [raw[i] for i in range(len(raw))]
    raise ValueError("data.yaml names must have contiguous numeric class indexes")


def _audit_root(root):
    errors = []
    yaml_files = sorted(root.rglob("data.yaml"))
    if len(yaml_files) != 1:
        raise ValueError("Expected exactly one data.yaml")
    base = yaml_files[0].parent
    data = yaml.safe_load(yaml_files[0].read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid data.yaml")
    provenance = data.get("roboflow")
    if not isinstance(provenance, dict) or provenance.get("workspace") != "joshuas-workspace" or provenance.get("project") != "playing-cards-9gfac" or str(provenance.get("version")) != "2":
        raise ValueError("Export is not Joshuas Workspace / playing-cards-9gfac version 2")
    names = _names(data.get("names"))
    if len(names) != 52 or len(set(names)) != 52 or set(names) != set(LABELS[:-1]):
        raise ValueError("Export class indexes must map to exactly the canonical 52 faces, without BACK")
    if data.get("nc", 52) != 52:
        raise ValueError("data.yaml nc differs from 52 names")
    counts = Counter()
    hashes = defaultdict(list)
    annotations = 0
    boxes = 0
    for split in ("train", "valid", "val", "test"):
        images = base / split / "images"
        labels = base / split / "labels"
        if not images.exists():
            continue
        for image in sorted(p for p in images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
            if image.is_symlink() or not image.resolve().is_relative_to(base.resolve()):
                errors.append(f"Image escapes export root: {image.relative_to(base)}")
                continue
            relative = image.relative_to(images)
            label = labels / relative.with_suffix(".txt")
            if not label.is_file() or label.is_symlink():
                errors.append(f"Missing annotation: {label.relative_to(base)}")
                continue
            try:
                with Image.open(image) as raster:
                    width, height = raster.size
                    raster.verify()
                if width <= 0 or height <= 0:
                    raise ValueError("empty image")
                for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    tokens = line.split()
                    if len(tokens) != 5 or not re.fullmatch(r"\d+", tokens[0]):
                        raise ValueError(f"Invalid YOLO row {line_no}")
                    class_id = int(tokens[0])
                    if not 0 <= class_id < 52:
                        raise ValueError(f"Unknown class index on row {line_no}")
                    import math
                    cx, cy, w, h = map(float, tokens[1:])
                    if not all(map(math.isfinite, (cx, cy, w, h))) or w <= 0 or h <= 0 or cx-w/2 < -1e-6 or cy-h/2 < -1e-6 or cx+w/2 > 1+1e-6 or cy+h/2 > 1+1e-6:
                        raise ValueError(f"Invalid box on row {line_no}")
                    boxes += 1
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                hashes[digest].append(image.relative_to(base).as_posix())
                counts[split] += 1
                annotations += 1
            except (OSError, UnicodeError, ValueError) as error:
                errors.append(f"{image.relative_to(base)}: {error}")
        referenced = {p.relative_to(images).with_suffix(".txt") for p in images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
        for orphan in sorted(p for p in labels.rglob("*.txt") if p.relative_to(labels) not in referenced):
            errors.append(f"Orphan annotation: {orphan.relative_to(base)}")
    if not counts:
        errors.append("No annotated images")
    return {
        "artifact_status": "INVALID" if errors else "SOURCE_ARTIFACT_VALID",
        "training_status": "NOT READY",
        "source_id": SOURCE_ID,
        "class_index_to_label": {str(i): label for i, label in enumerate(names)},
        "image_count": sum(counts.values()),
        "annotation_count": annotations,
        "box_count": boxes,
        "upstream_split_images": dict(sorted(counts.items())),
        "exact_duplicate_files": [sorted(paths) for paths in hashes.values() if len(paths) > 1],
        "grouping_verified": False,
        "errors": sorted(errors),
    }


def audit_artifact(path, *, expected_source_id=SOURCE_ID, archive_sha256=None):
    path = Path(path)
    if expected_source_id != SOURCE_ID:
        return {"artifact_status": "INVALID", "training_status": "NOT READY", "errors": ["Wrong source ID"]}
    try:
        if path.is_file():
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if archive_sha256 and actual_hash != archive_sha256:
                raise ValueError("Archive SHA-256 mismatch")
            with tempfile.TemporaryDirectory() as directory, zipfile.ZipFile(path) as archive:
                root = Path(directory)
                for member in archive.infolist():
                    target = root / member.filename
                    if member.filename.startswith("/") or not target.resolve().is_relative_to(root.resolve()):
                        raise ValueError("Unsafe archive path")
                    if member.file_size > 512 * 1024 * 1024:
                        raise ValueError("Oversized archive member")
                archive.extractall(root)
                report = _audit_root(root)
            report["archive_sha256"] = actual_hash
            return report
        report = _audit_root(path)
        report["archive_sha256"] = None
        return report
    except (OSError, ValueError, zipfile.BadZipFile, yaml.YAMLError) as error:
        return {"artifact_status": "INVALID", "training_status": "NOT READY", "source_id": SOURCE_ID,
                "errors": [str(error)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--source-id", default=SOURCE_ID)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_artifact(args.path, expected_source_id=args.source_id, archive_sha256=args.archive_sha256)
    value = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(value, encoding="utf-8")
    print(value, end="")
    raise SystemExit(0 if report["artifact_status"] == "SOURCE_ARTIFACT_VALID" else 2)


if __name__ == "__main__":
    main()
