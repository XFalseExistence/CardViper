"""Strict B1 crop-manifest loader; TensorFlow remains optional."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re

from PIL import Image

from .labels import LABELS
from .manifest import local_path
from .build_classifier_crops import crop_bounds


@dataclass(frozen=True)
class CropSample:
    crop_path: Path
    label: str
    label_index: int
    source_id: str
    scene_id: str
    session_id: str | None
    source_image_sha256: str
    annotation_id: str = ""


def load_crop_split(directory, split):
    if split not in ("train", "val", "test", "holdout"):
        raise ValueError("Unknown split")
    directory = Path(directory)
    samples = []
    for line_number, line in enumerate((directory / "crops.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            for key in ("source_id", "scene_id", "source_image", "source_image_sha256", "source_annotation", "annotation_id", "crop_path", "label"):
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError(f"Invalid {key}")
            if not re.fullmatch(r"[0-9a-f]{64}", row["source_image_sha256"]):
                raise ValueError("Malformed source image SHA-256 provenance")
            if row.get("schema_version") != 1 or type(row.get("held_out")) is not bool:
                raise ValueError("Malformed crop provenance")
            # B1 reserves a whole connected scene/session group for holdout;
            # only the triggering record must carry held_out=True.
            if split != "holdout" and row["held_out"]:
                raise ValueError("Holdout provenance conflicts with split")
            label = row["label"]
            if label not in LABELS:
                raise ValueError("Unknown label")
            index = LABELS.index(label)
            if "label_index" in row and row["label_index"] != index:
                raise ValueError("Label index drift")
            path = local_path(directory, row["crop_path"])
            if not path.is_file():
                raise ValueError("Missing crop file")
            with Image.open(path) as image:
                image.verify()
            samples.append(CropSample(path, label, index, row["source_id"], row["scene_id"],
                                      row.get("session_id"), row["source_image_sha256"], row["annotation_id"]))
        except (KeyError, ValueError, OSError) as error:
            raise ValueError(f"{directory}/crops.jsonl:{line_number}: {error}") from error
    return tuple(samples)


def seeded_batches(samples, batch_size, seed):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    items = list(samples)
    random.Random(seed).shuffle(items)
    return tuple(tuple(items[i:i + batch_size]) for i in range(0, len(items), batch_size))


def load_all_splits(crop_root):
    splits = {name: load_crop_split(Path(crop_root) / name, name) for name in ("train", "val", "test", "holdout")}
    ownership = {}
    for name, samples in splits.items():
        for sample in samples:
            keys = [(sample.source_id, "scene", sample.scene_id), ("hash", sample.source_image_sha256)]
            if sample.session_id:
                keys.append((sample.source_id, "session", sample.session_id))
            for key in keys:
                if key in ownership and ownership[key] != name:
                    raise ValueError(f"Crop leakage across splits: {key}")
                ownership[key] = name
    return splits


def verify_crop_provenance(directory, split_rows, roots, split):
    """Match every B1 crop row and pixel to its approved source annotation."""
    directory = Path(directory)
    expected = {}
    for source in split_rows:
        for obj in source.objects:
            key = (source.source_id, source.image_sha256, obj.annotation_id, obj.label)
            if key in expected:
                raise ValueError("Duplicate approved crop annotation")
            expected[key] = (source, obj)
    seen = set()
    for number, line in enumerate((directory / "crops.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        key = (row.get("source_id"), row.get("source_image_sha256"), row.get("annotation_id"), row.get("label"))
        if key not in expected or key in seen:
            raise ValueError(f"Crop coverage differs from approved {split} annotations at row {number}")
        seen.add(key)
        source, obj = expected[key]
        fixed = {"schema_version": 1, "source_id": source.source_id, "scene_id": source.scene_id,
                 "session_id": source.session_id, "held_out": source.held_out,
                 "source_image": source.image_path, "source_image_sha256": source.image_sha256,
                 "source_width": source.width, "source_height": source.height,
                 "source_annotation": source.annotation_path, "annotation_id": obj.annotation_id,
                 "source_bbox": list(obj.bbox), "label": obj.label}
        if any(row.get(field) != value for field, value in fixed.items()):
            raise ValueError(f"Crop provenance mismatch at row {number}")
        bounds = crop_bounds(obj.bbox, source.width, source.height, row.get("padding"))
        if row.get("crop_bbox") != list(bounds):
            raise ValueError(f"Crop geometry mismatch at row {number}")
        unsigned = {k: v for k, v in row.items() if k != "crop_path"}
        filename = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() + ".png"
        if row.get("crop_path") != filename:
            raise ValueError(f"Crop filename/provenance hash mismatch at row {number}")
        if source.source_id not in roots:
            raise ValueError(f"Missing root for {source.source_id}")
        source_path = local_path(roots[source.source_id], source.image_path)
        crop_path = local_path(directory, filename)
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != source.image_sha256:
            raise ValueError(f"Source image hash mismatch at row {number}")
        with Image.open(source_path) as original, Image.open(crop_path) as actual:
            if original.size != (source.width, source.height):
                raise ValueError(f"Source dimensions changed at row {number}")
            expected_crop = original.convert("RGB").crop(bounds)
            if actual.mode != "RGB" or actual.size != expected_crop.size or actual.tobytes() != expected_crop.tobytes():
                raise ValueError(f"Crop pixels differ from approved source at row {number}")
    if seen != set(expected):
        raise ValueError(f"Crop coverage differs from approved {split} annotations")
    return load_crop_split(directory, split)


def rgb_array(sample, input_size):
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    with Image.open(sample.crop_path) as image:
        return image.convert("RGB").resize((input_size, input_size), Image.Resampling.BILINEAR)
