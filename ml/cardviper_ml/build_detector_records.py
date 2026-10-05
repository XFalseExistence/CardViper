"""Lossless scene provenance with exactly one detector semantic class, CARD."""

import argparse
import json
from pathlib import Path

from .manifest import ImageRecord, read_manifest
from .splits import group_keys

DETECTOR_CLASSES = ("CARD",)


def build_detector_records(records):
    output = []
    for row in records:
        if not isinstance(row, ImageRecord):
            raise ValueError("Expected validated ImageRecord; unannotated data is not a negative")
        output.append({
            "schema_version": 1,
            "source_id": row.source_id, "scene_id": row.scene_id, "session_id": row.session_id,
            "image_path": row.image_path, "image_sha256": row.image_sha256,
            "width": row.width, "height": row.height,
            "annotation_path": row.annotation_path, "held_out": row.held_out,
            "group_keys": [list(key) for key in group_keys(row)],
            "objects": [{
                "annotation_id": obj.annotation_id, "bbox": list(obj.bbox),
                "class_id": 0, "class_name": "CARD",
                "source_annotation_type": obj.source_annotation_type,
                "source_annotation_path": obj.source_annotation_path,
                "source_annotation_row": obj.source_annotation_row,
                "polygon_point_count": obj.polygon_point_count,
                "source_annotation_sha256": obj.source_annotation_sha256,
            } for obj in row.objects],
        })
    return output


def write_detector_records(path, records):
    rows = build_detector_records(records)
    Path(path).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                          encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    write_detector_records(args.output, rows)
    print(json.dumps({"class_names": DETECTOR_CLASSES, "scene_count": len(rows),
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
