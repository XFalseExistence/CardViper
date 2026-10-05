"""Import explicitly confirmed, CardViper-owned no-card scenes as detector negatives."""

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

from .manifest import ImageRecord, local_path, nonempty, write_manifest

SOURCE_ID = "cardviper-detector-negatives"


def import_negatives(root):
    root = Path(root).resolve()
    manifest = root / "scenes.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("source_id") != SOURCE_ID:
        raise ValueError("Expected CardViper detector-negative schema version 1")
    nonempty(data.get("owner"), "owner")
    rights = data.get("rights")
    if not isinstance(rights, dict) or rights.get("detector") is not True:
        raise ValueError("Explicit detector rights are required")
    nonempty(rights.get("evidence"), "rights evidence")
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scenes must be a nonempty list")
    rows, seen = [], set()
    for index, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict) or "objects" in scene:
            raise ValueError(f"scene {index}: use explicit no_card_confirmed, not objects")
        for key in ("scene_id", "session_id", "capture_date", "device",
                    "image_path", "annotation_provenance"):
            nonempty(scene.get(key), key)
        try:
            if date.fromisoformat(scene["capture_date"]).isoformat() != scene["capture_date"]:
                raise ValueError("noncanonical date")
        except ValueError as error:
            raise ValueError(f"scene {index}: capture_date must be YYYY-MM-DD") from error
        if scene.get("no_card_confirmed") is not True:
            raise ValueError(f"scene {index}: no_card_confirmed must be explicitly true")
        if type(scene.get("held_out")) is not bool:
            raise ValueError(f"scene {index}: held_out must be explicitly boolean")
        if scene["image_path"] in seen:
            raise ValueError(f"scene {index}: duplicate image path")
        seen.add(scene["image_path"])
        image = local_path(root, scene["image_path"])
        if not image.is_file():
            raise ValueError(f"scene {index}: image missing")
        digest = scene.get("image_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"scene {index}: invalid image SHA-256")
        if hashlib.sha256(image.read_bytes()).hexdigest() != digest:
            raise ValueError(f"scene {index}: image SHA-256 mismatch")
        with Image.open(image) as raster:
            width, height = raster.size
            raster.verify()
        rows.append(ImageRecord(source_id=SOURCE_ID, scene_id=scene["scene_id"],
                                session_id=scene["session_id"], image_path=scene["image_path"],
                                image_sha256=digest, width=width, height=height,
                                annotation_path="scenes.json", objects=(),
                                held_out=scene["held_out"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = import_negatives(args.root)
    write_manifest(args.output, rows)
    print(json.dumps({"source_id": SOURCE_ID, "negative_scenes": len(rows),
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
