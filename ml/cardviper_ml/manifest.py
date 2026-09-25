"""Version 1 JSONL: one source image and zero or more exact card annotations.

Boxes are (left, top, right, bottom) in source pixels, half-open, fully inside
stored raster dimensions. No EXIF transposition is performed. Empty objects
means an explicitly annotated no-card negative, not an unlabelled image.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path, PurePosixPath
import re

from .labels import decode


def nonempty(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, unpadded string")


def relative_path(value):
    nonempty(value, "path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value or path.as_posix() != value or value == ".":
        raise ValueError("Expected a normalized relative POSIX path")


def local_path(root: Path, relative: str) -> Path:
    relative_path(relative)
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes source root (including symlinks)")
    return path


def checked_box(box, width=None, height=None):
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        raise ValueError("bbox must have four coordinates")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
        raise ValueError("bbox coordinates must be finite numbers")
    left, top, right, bottom = box
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError("Degenerate or malformed bbox")
    if width is not None and (right > width or bottom > height):
        raise ValueError("bbox outside source image")
    return tuple(box)


@dataclass(frozen=True)
class Annotation:
    annotation_id: str
    label: str
    bbox: tuple[float, float, float, float]
    detector_class: str = "CARD"

    def __post_init__(self):
        nonempty(self.annotation_id, "annotation_id")
        decode(self.label)
        object.__setattr__(self, "bbox", checked_box(self.bbox))
        if self.detector_class != "CARD":
            raise ValueError("Detector semantic class must be CARD")


@dataclass(frozen=True)
class ImageRecord:
    source_id: str
    scene_id: str
    image_path: str
    width: int
    height: int
    image_sha256: str
    annotation_path: str
    objects: tuple[Annotation, ...]
    session_id: str | None = None
    held_out: bool = False
    schema_version: int = 1

    def __post_init__(self):
        nonempty(self.source_id, "source_id")
        nonempty(self.scene_id, "scene_id")
        if self.session_id is not None:
            nonempty(self.session_id, "session_id")
        relative_path(self.image_path)
        relative_path(self.annotation_path)
        if any(type(v) is not int or v <= 0 for v in (self.width, self.height)):
            raise ValueError("Source dimensions must be positive integers")
        if not isinstance(self.image_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.image_sha256):
            raise ValueError("image_sha256 must be a lowercase SHA-256 digest")
        if type(self.held_out) is not bool or type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Invalid held_out flag or schema version")
        object.__setattr__(self, "objects", tuple(self.objects))
        ids, boxes = set(), set()
        for obj in self.objects:
            if not isinstance(obj, Annotation):
                raise ValueError("Expected Annotation objects")
            checked_box(obj.bbox, self.width, self.height)
            if obj.annotation_id in ids or obj.bbox in boxes:
                raise ValueError("Duplicate annotation id or box; resolve annotations explicitly")
            ids.add(obj.annotation_id)
            boxes.add(obj.bbox)


def record_from_dict(data):
    try:
        value = dict(data)
        value["objects"] = tuple(Annotation(**obj) for obj in value["objects"])
        return ImageRecord(**value)
    except (TypeError, KeyError) as error:
        raise ValueError(f"Invalid manifest record: {error}") from error


def read_manifest(path: Path) -> list[ImageRecord]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(record_from_dict(json.loads(line)))
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return rows


def write_manifest(path: Path, records) -> None:
    Path(path).write_text("".join(json.dumps(asdict(r), sort_keys=True, allow_nan=False) + "\n" for r in records), encoding="utf-8")


def read_sources(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unknown source manifest version")
    sources = {}
    for source in data["sources"]:
        for key in ("source_id", "name", "local_root", "notes"):
            nonempty(source.get(key), key)
        if source["source_id"] in sources:
            raise ValueError("Duplicate source_id")
        if type(source.get("verified")) is not bool:
            raise ValueError("Source verification must be explicit")
        permitted = source.get("permitted_use", {})
        if set(permitted) != {"detector", "classifier"} or any(type(v) is not bool for v in permitted.values()):
            raise ValueError("Both training permissions must be explicit booleans")
        if source["verified"]:
            for key in ("url", "license", "attribution"):
                nonempty(source.get(key), key)
        elif any(permitted.values()):
            raise ValueError("Unverified source cannot permit training")
        sources[source["source_id"]] = source
    return sources


def require_training_permission(sources, source_id, purpose):
    source = sources.get(source_id)
    if not source or not source["verified"]:
        raise ValueError(f"Source {source_id!r} is not verified for training")
    if not source["permitted_use"].get(purpose, False):
        raise ValueError(f"Source {source_id!r} does not permit {purpose} training")
