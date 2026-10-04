"""Import and audit local CardViper-owned BACK captures without changing images."""

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

from .manifest import Annotation, ImageRecord, local_path, nonempty, write_manifest
from .splits import connected_groups

SOURCE_ID = "cardviper-back"
MODES = {"tight-back-crop", "annotated-scene"}


def _capture_data(root, capture_mode):
    root = Path(root).resolve()
    if capture_mode is not None and capture_mode not in MODES:
        raise ValueError("Unknown capture_mode")
    manifest_path = root / "captures.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("source_id") != SOURCE_ID:
        raise ValueError("Expected schema_version 1 and source_id cardviper-back")
    nonempty(data.get("owner"), "owner")
    rights = data.get("rights")
    if not isinstance(rights, dict) or any(type(rights.get(key)) is not bool for key in ("classifier", "detector")):
        raise ValueError("rights must explicitly state classifier and detector permission")
    nonempty(rights.get("evidence"), "rights evidence")
    if rights["detector"]:
        nonempty(rights.get("detector_evidence"), "detector_evidence")
    captures = data.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("captures must be a nonempty list")
    return root, manifest_path, data


def _load(root, capture_mode=None):
    root, manifest_path, data = _capture_data(root, capture_mode)
    rows, metadata = [], []
    seen_paths = set()
    for index, entry in enumerate(data["captures"], 1):
        if not isinstance(entry, dict):
            raise ValueError(f"capture {index}: expected an object")
        for key in ("capture_date", "device", "session_id", "scene_id", "deck_design",
                    "annotation_provenance", "image_path"):
            nonempty(entry.get(key), key)
        try:
            if date.fromisoformat(entry["capture_date"]).isoformat() != entry["capture_date"]:
                raise ValueError("date must be YYYY-MM-DD")
        except ValueError as error:
            raise ValueError(f"capture {index}: invalid capture_date") from error
        mode = entry.get("capture_mode")
        if mode not in MODES or (capture_mode is not None and mode != capture_mode):
            raise ValueError(f"capture {index}: capture_mode must match the declared mode")
        image_path = entry["image_path"]
        path = local_path(root, image_path)
        if image_path in seen_paths:
            raise ValueError(f"capture {index}: duplicate image_path")
        seen_paths.add(image_path)
        if not path.is_file():
            raise ValueError(f"capture {index}: missing image at {image_path}")
        digest = entry.get("image_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"capture {index}: image SHA-256 must be lowercase hex")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"capture {index}: image SHA-256 mismatch")
        with Image.open(path) as raster:
            width, height = raster.size
            raster.verify()
        if mode == "tight-back-crop":
            if entry.get("full_image_back") is not True:
                raise ValueError(f"capture {index}: full_image_back must explicitly be true")
            if entry.get("boxes"):
                raise ValueError(f"capture {index}: tight-back-crop cannot also have boxes")
            objects = (Annotation(f"back-{index}", "BACK", (0, 0, width, height)),)
        else:
            if "full_image_back" in entry:
                raise ValueError(f"capture {index}: annotated-scene cannot declare full_image_back")
            boxes = entry.get("boxes")
            if not isinstance(boxes, list) or not boxes:
                raise ValueError(f"capture {index}: annotated-scene requires real boxes")
            objects = []
            for box in boxes:
                if not isinstance(box, dict) or box.get("label") != "BACK":
                    raise ValueError(f"capture {index}: every box needs label BACK")
                try:
                    objects.append(Annotation(box["annotation_id"], box["label"], box["bbox"]))
                except KeyError as error:
                    raise ValueError(f"capture {index}: box missing {error.args[0]}") from error
            objects = tuple(objects)
        held_out = entry.get("held_out", False)
        rows.append(ImageRecord(source_id=SOURCE_ID, scene_id=entry["scene_id"],
                                session_id=entry["session_id"], held_out=held_out,
                                image_path=image_path, width=width, height=height,
                                image_sha256=digest, annotation_path="captures.json",
                                objects=objects))
        metadata.append({"deck_design": entry["deck_design"], "capture_mode": mode,
                         "capture_date": entry["capture_date"], "device": entry["device"]})
    return rows, metadata, data, manifest_path, root


def import_back_source(root, *, capture_mode=None) -> list[ImageRecord]:
    """Return normalized BACK rows from an explicitly annotated local batch."""
    return _load(root, capture_mode)[0]


def _difference_hash(path):
    with Image.open(path) as raster:
        reduced = raster.convert("L").resize((9, 8))
        pixels = [reduced.getpixel((x, y)) for y in range(8) for x in range(9)]
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return bits


def audit_back_source(root, *, capture_mode=None) -> dict:
    """Report structural split eligibility and separately gated human review."""
    rows, metadata, data, manifest_path, root = _load(root, capture_mode)
    digests = defaultdict(list)
    for row in rows:
        digests[row.image_sha256].append(row.image_path)
    duplicates = sorted(sorted(paths) for paths in digests.values() if len(paths) > 1)
    hashes = [_difference_hash(local_path(root, row.image_path)) for row in rows]
    near = []
    for i, left in enumerate(rows):
        for j in range(i + 1, len(rows)):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance <= 6:
                near.append({"paths": [left.image_path, rows[j].image_path],
                             "dhash_distance": distance})
    sessions = Counter(row.session_id for row in rows)
    designs = Counter(item["deck_design"] for item in metadata)
    dimensions = Counter(f"{row.width}x{row.height}" for row in rows)
    modes = Counter(item["capture_mode"] for item in metadata)
    groups = connected_groups(rows)
    independent = [group for group in groups if not any(row.held_out for row in group)]
    reasons = []
    if not data["rights"]["classifier"]:
        reasons.append("Capture rights do not permit classifier use")
    if duplicates:
        reasons.append("Exact image duplicates must be reconciled before splitting")
    if len(independent) < 3:
        reasons.append("Need at least three independent non-held-out scene/session groups")
    split_eligibility = "READY" if not reasons else "NOT READY"
    if len(designs) < 2:
        reasons.append("Need captures of more than one physical deck-back design")
    review = data.get("review")
    if not (isinstance(review, dict) and review.get("approved") is True
            and review.get("session_independence_confirmed") is True
            and review.get("near_duplicate_clues_reviewed") is True
            and isinstance(review.get("reviewer"), str) and review["reviewer"].strip()
            and isinstance(review.get("notes"), str) and review["notes"].strip()):
        reasons.append("Human provenance, diversity and similarity review is required")
    return {
        "source_id": SOURCE_ID, "source_root": str(root),
        "capture_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "owner": data["owner"], "rights": data["rights"],
        "sample_count": len(rows), "annotation_count": sum(len(row.objects) for row in rows),
        "session_count": len(sessions), "independent_sessions": len(independent),
        "connected_group_count": len(groups), "design_count": len(designs),
        "session_distribution": dict(sorted(sessions.items())),
        "design_distribution": dict(sorted(designs.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "capture_modes": dict(sorted(modes.items())),
        "exact_duplicates": duplicates, "near_duplicate_clues": near,
        "split_eligibility": split_eligibility,
        "status": "READY" if not reasons else "NOT READY", "reasons": reasons,
        "review": review,
        "qualification": "Counts and perceptual hash clues do not prove visual diversity or independent sessions; READY requires an explicit human review.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Ignored local capture root containing captures.json")
    parser.add_argument("--capture-mode", choices=sorted(MODES))
    parser.add_argument("--output", type=Path, help="Normalized ImageRecord JSONL output")
    parser.add_argument("--report", type=Path, help="Audit JSON output")
    args = parser.parse_args()
    rows = import_back_source(args.root, capture_mode=args.capture_mode)
    report = audit_back_source(args.root, capture_mode=args.capture_mode)
    if args.output:
        write_manifest(args.output, rows)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
