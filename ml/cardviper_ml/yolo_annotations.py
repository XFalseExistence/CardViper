"""Strict YOLO detection and polygon rows, normalized to enclosing boxes."""

from dataclasses import dataclass
import hashlib
import math
import re


@dataclass(frozen=True)
class ParsedYoloRow:
    class_id: int
    bbox: tuple[float, float, float, float]
    annotation_type: str
    point_count: int | None
    source_sha256: str


def parse_yolo_row(line: str, class_count: int) -> ParsedYoloRow:
    tokens = line.split()
    if not tokens or not re.fullmatch(r"\d+", tokens[0]):
        raise ValueError("Expected a nonnegative integer class index")
    class_id = int(tokens[0])
    if not 0 <= class_id < class_count:
        raise ValueError("Unknown class index")
    try:
        values = tuple(map(float, tokens[1:]))
    except ValueError as error:
        raise ValueError("Invalid YOLO coordinate") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("YOLO coordinates must be finite")
    if len(values) == 4:
        cx, cy, width, height = values
        left, top, right, bottom = (cx - width / 2, cy - height / 2,
                                    cx + width / 2, cy + height / 2)
        if width <= 0 or height <= 0 or left < 0 or top < 0 or right > 1 or bottom > 1:
            raise ValueError("Invalid normalized YOLO box")
        kind, point_count = "bbox", None
    else:
        if len(values) < 6 or len(values) % 2:
            raise ValueError("Polygon requires at least three coordinate pairs")
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("Polygon coordinates must be normalized to 0..1")
        xs, ys = values[0::2], values[1::2]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        if right <= left or bottom <= top:
            raise ValueError("Polygon has zero extent")
        kind, point_count = "polygon", len(xs)
    return ParsedYoloRow(class_id, (left, top, right, bottom), kind, point_count,
                         hashlib.sha256(line.encode("utf-8")).hexdigest())
