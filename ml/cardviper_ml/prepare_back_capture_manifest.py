"""Hash session-folder BACK photos into a deliberately unapproved local draft."""

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

from PIL import Image

from .import_back_captures import MODES, SOURCE_ID
from .manifest import nonempty

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _exif_date(raster):
    raw = raster.getexif().get(36867) or raster.getexif().get(306)
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw[:10].replace(":", "-")).isoformat()
        except ValueError:
            pass
    return ""


def prepare_manifest(root, *, device, capture_mode):
    root = Path(root).resolve()
    nonempty(device, "device")
    if capture_mode not in MODES:
        raise ValueError("capture_mode must be tight-back-crop or annotated-scene")
    if not root.is_dir():
        raise ValueError("Capture root must exist")
    target = root / "captures.json"
    if target.exists():
        raise FileExistsError(f"Preserving existing manifest: {target}")
    captures = []
    for image in sorted(root.rglob("*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image.relative_to(root)
        if (len(relative.parts) != 2 or image.is_symlink() or
                not image.resolve().is_relative_to(root)):
            raise ValueError(f"Use one session directory per image, without symlinks: {relative}")
        with Image.open(image) as raster:
            width, height = raster.size
            capture_date = _exif_date(raster)
        with Image.open(image) as raster:
            raster.verify()
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {relative}")
        image_path = relative.as_posix()
        capture = {
            "image_path": image_path,
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "width": width, "height": height,
            "capture_date": capture_date,
            "device": device,
            "session_id": relative.parts[0],
            "scene_id": "back-" + hashlib.sha256(image_path.encode()).hexdigest()[:16],
            "deck_design": "",
            "annotation_provenance": "",
            "capture_mode": capture_mode,
        }
        if capture_mode == "tight-back-crop":
            capture["full_image_back"] = False
        else:
            capture["boxes"] = []
        captures.append(capture)
    if not captures:
        raise ValueError("No images found under session directories")
    payload = {
        "schema_version": 1, "source_id": SOURCE_ID, "draft_status": "NOT READY",
        "owner": "",
        "rights": {"classifier": False, "detector": False, "evidence": ""},
        "captures": captures,
        "review": {"approved": False, "session_independence_confirmed": False,
                   "near_duplicate_clues_reviewed": False, "reviewer": "", "notes": ""},
    }
    with target.open("x", encoding="utf-8") as output:
        json.dump(payload, output, sort_keys=True, indent=2)
        output.write("\n")
    return {"status": "DRAFT", "manifest": str(target), "sample_count": len(captures),
            "session_count": len({item["session_id"] for item in captures}),
            "required_review": ["capture_date", "deck_design", "owner", "rights.evidence",
                                "annotation_provenance", "full_image_back or boxes",
                                "session independence", "near-duplicate clues", "review approval"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--capture-mode", required=True, choices=sorted(MODES))
    args = parser.parse_args()
    print(json.dumps(prepare_manifest(args.root, device=args.device,
                                      capture_mode=args.capture_mode), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
