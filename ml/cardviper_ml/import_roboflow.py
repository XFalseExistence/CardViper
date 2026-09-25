"""Normalize LOCAL Roboflow YOLO detection exports; never download anything."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

from .labels import decode
from .manifest import Annotation, ImageRecord, local_path, read_sources, write_manifest


def import_yolo(root, source_id, class_names, groups, *, class_map=None,
                images="images", annotations="labels") -> list[ImageRecord]:
    root = Path(root).resolve()
    if not isinstance(class_names, list) or not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Provide a nonempty, unique export class-index list")
    mapping = class_map or {}
    if set(mapping) - set(class_names):
        raise ValueError("Alias map includes unknown export classes")
    labels = [mapping.get(name, name) for name in class_names]
    for label in labels:
        decode(label)
    if len(set(labels)) != len(labels):
        raise ValueError("Export classes map to duplicate identities")
    image_dir = local_path(root, images)
    label_dir = local_path(root, annotations)
    paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    if not paths:
        raise ValueError("No source images found")
    rows = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        path = local_path(root, relative)
        group = groups.get(relative)
        if not isinstance(group, dict) or not group.get("scene_id"):
            raise ValueError(f"Missing scene group for {relative}")
        annotation = label_dir / Path(relative).relative_to(images).with_suffix(".txt")
        annotation_relative = annotation.relative_to(root).as_posix()
        annotation = local_path(root, annotation_relative)
        if not annotation.is_file():
            raise ValueError(f"Missing annotation file: {annotation_relative}; not assumed negative")
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        objects = []
        for line_number, line in enumerate(annotation.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            tokens = line.split()
            try:
                if len(tokens) != 5 or not re.fullmatch(r"\d+", tokens[0]):
                    raise ValueError("Expected class_id center_x center_y width height")
                class_id = int(tokens[0])
                if not 0 <= class_id < len(labels):
                    raise ValueError("Unknown class index")
                cx, cy, w, h = map(float, tokens[1:])
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError("YOLO coordinates must be finite and normalized to 0..1")
                box = ((cx - w / 2) * width, (cy - h / 2) * height,
                       (cx + w / 2) * width, (cy + h / 2) * height)
                objects.append(Annotation(str(line_number), labels[class_id], box))
            except ValueError as error:
                raise ValueError(f"{annotation_relative}:{line_number}: {error}") from error
        rows.append(ImageRecord(source_id=source_id, scene_id=group["scene_id"],
                                session_id=group.get("session_id"), held_out=group.get("held_out", False),
                                image_path=relative, width=width, height=height,
                                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                annotation_path=annotation_relative, objects=tuple(objects)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--classes", required=True, type=Path, help="JSON list in original export index order")
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--class-map", type=Path, help="Explicit aliases JSON, e.g. {\"ac\": \"AC\"}")
    parser.add_argument("--images", default="images")
    parser.add_argument("--annotations", default="labels")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.source_id not in read_sources(args.sources):
        parser.error("source-id must be registered in the source manifest")
    rows = import_yolo(args.root, args.source_id, json.loads(args.classes.read_text()),
                       json.loads(args.groups.read_text()),
                       class_map=json.loads(args.class_map.read_text()) if args.class_map else None,
                       images=args.images, annotations=args.annotations)
    write_manifest(args.output, rows)


if __name__ == "__main__":
    main()
