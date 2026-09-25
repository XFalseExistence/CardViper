"""Source-resolution RGB classifier crops, with immutable source provenance."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from .manifest import checked_box, local_path, read_manifest
from .splits import record_key, unique_records


def crop_bounds(box, width, height, padding=0.0):
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("Source dimensions must be positive integers")
    if type(padding) not in (int, float) or not math.isfinite(padding) or not 0 <= padding <= 0.5:
        raise ValueError("padding must be finite in 0..0.5 per side")
    left, top, right, bottom = checked_box(box, width, height)
    dx, dy = (right - left) * padding, (bottom - top) * padding
    bounds = (max(0, math.floor(left - dx)), max(0, math.floor(top - dy)),
              min(width, math.ceil(right + dx)), min(height, math.ceil(bottom + dy)))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Empty bbox after padding/clamping")
    return bounds


def build_crops(records, roots, output, *, padding=0.0):
    rows = sorted(records, key=record_key)
    unique_records(rows)
    # Validate even an all-negative dataset's padding configuration.
    crop_bounds((0, 0, 1, 1), 1, 1, padding)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    crops = []
    for row in rows:
        if row.source_id not in roots:
            raise ValueError(f"Missing local root for source {row.source_id!r}")
        source = local_path(roots[row.source_id], row.image_path)
        if hashlib.sha256(source.read_bytes()).hexdigest() != row.image_sha256:
            raise ValueError(f"Source hash changed: {row.source_id}/{row.image_path}")
        with Image.open(source) as original:
            if original.size != (row.width, row.height):
                raise ValueError(f"Source dimensions changed: {row.image_path}")
            # Use stored raster geometry, not EXIF-transposed geometry.
            image = original.convert("RGB")
            try:
                for obj in row.objects:
                    bounds = crop_bounds(obj.bbox, row.width, row.height, padding)
                    crop = {
                        "schema_version": 1,
                        "source_id": row.source_id, "scene_id": row.scene_id,
                        "session_id": row.session_id, "held_out": row.held_out,
                        "source_image": row.image_path, "source_image_sha256": row.image_sha256,
                        "source_width": row.width, "source_height": row.height,
                        "source_annotation": row.annotation_path, "annotation_id": obj.annotation_id,
                        "source_bbox": list(obj.bbox), "label": obj.label,
                        "crop_bbox": list(bounds), "padding": padding,
                    }
                    name = hashlib.sha256(json.dumps(crop, sort_keys=True).encode()).hexdigest() + ".png"
                    crop["crop_path"] = name
                    with image.crop(bounds) as cropped:
                        cropped.save(output / name, format="PNG")
                    crops.append(crop)
            finally:
                image.close()
    # Written only after every crop succeeds: partial runs have no usable manifest.
    (output / "crops.jsonl").write_text(
        "".join(json.dumps(crop, sort_keys=True, allow_nan=False) + "\n" for crop in crops), encoding="utf-8")
    return crops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--roots", required=True, type=Path, help="Local JSON source_id to absolute root path map")
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--padding", type=float, default=0.0)
    args = parser.parse_args()
    crops = build_crops(read_manifest(args.manifest), json.loads(args.roots.read_text()), args.output, padding=args.padding)
    print(f"Generated {len(crops)} crops; provenance: {args.output / 'crops.jsonl'}")


if __name__ == "__main__":
    main()
