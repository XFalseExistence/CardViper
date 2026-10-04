"""Strict B1 crop-manifest loader; TensorFlow remains optional."""

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re

from PIL import Image

from .labels import LABELS
from .manifest import local_path


@dataclass(frozen=True)
class CropSample:
    crop_path: Path
    label: str
    label_index: int
    source_id: str
    scene_id: str
    session_id: str | None
    source_image_sha256: str


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
            if row["held_out"] != (split == "holdout"):
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
                                      row.get("session_id"), row["source_image_sha256"]))
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


def rgb_array(sample, input_size):
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    with Image.open(sample.crop_path) as image:
        return image.convert("RGB").resize((input_size, input_size), Image.Resampling.BILINEAR)
