"""Conservative image-family clues; only explicit provenance can clear grouping."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from PIL import Image


def filename_family(path):
    stem = path.rsplit("/", 1)[-1]
    stem = re.sub(r"(?i)\.(jpg|jpeg|png|webp|bmp)$", "", stem)
    stem = re.sub(r"(?i)_(jpg|jpeg|png|webp|bmp)\.rf\.[a-f0-9]+$", "", stem)
    return stem


def audit_families(rows):
    rows = sorted(rows, key=lambda row: row["path"])
    hashes, names, dimensions = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        hashes[row["sha256"]].append(row["path"])
        names[filename_family(row["path"])].append(row["path"])
        signature = (row["width"], row["height"], tuple(sorted(row["labels"])))
        dimensions[signature].append(row["path"])
    return {
        "exact_duplicates": sorted(sorted(paths) for paths in hashes.values() if len(paths) > 1),
        "probable_augmentation_families": sorted(sorted(paths) for paths in names.values() if len(paths) > 1),
        "unresolved_similarity_candidates": sorted(sorted(paths) for paths in dimensions.values() if len(paths) > 1),
        "orphan_images": [row["path"] for row in rows if row.get("annotation_exists") is False],
        "grouping_ready": False,
        "method": "SHA-256, filename, dimensions and annotation labels; clues require human provenance review",
    }


def collect_rows(root):
    """Read image/annotation evidence without assigning independent groups."""
    root = Path(root)
    rows = []
    for image in sorted(p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} and "images" in p.relative_to(root).parts):
        parts = image.relative_to(root).parts
        marker = parts.index("images")
        split_root = root.joinpath(*parts[:marker])
        nested_name = Path(*parts[marker + 1:]).with_suffix(".txt")
        label = split_root / "labels" / nested_name
        with Image.open(image) as raster:
            width, height = raster.size
            raster.verify()
        labels = []
        if label.is_file():
            labels = [line.split()[0] for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows.append({"path": image.relative_to(root).as_posix(),
                     "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                     "width": width, "height": height, "labels": labels,
                     "annotation_exists": label.is_file()})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_families(collect_rows(args.root))
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
