"""Continuous source/model box transforms; the actual exporter chooses resize policy later."""

from dataclasses import dataclass
import math

from .manifest import checked_box


@dataclass(frozen=True)
class ResizeTransform:
    source_width: int
    source_height: int
    model_width: int
    model_height: int
    policy: str = "letterbox"

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in
               (self.source_width, self.source_height, self.model_width, self.model_height)):
            raise ValueError("Source and model dimensions must be positive integers")
        if self.policy not in {"letterbox", "stretch"}:
            raise ValueError("Resize policy must be letterbox or stretch")

    @property
    def scale_x(self):
        if self.policy == "stretch":
            return self.model_width / self.source_width
        return min(self.model_width / self.source_width, self.model_height / self.source_height)

    @property
    def scale_y(self):
        return self.model_height / self.source_height if self.policy == "stretch" else self.scale_x

    @property
    def pad_x(self):
        return (self.model_width - self.source_width * self.scale_x) / 2

    @property
    def pad_y(self):
        return (self.model_height - self.source_height * self.scale_y) / 2

    def source_to_model(self, box):
        left, top, right, bottom = checked_box(box, self.source_width, self.source_height)
        return (left * self.scale_x + self.pad_x, top * self.scale_y + self.pad_y,
                right * self.scale_x + self.pad_x, bottom * self.scale_y + self.pad_y)

    def model_to_source(self, box, *, clamp=False):
        if (not isinstance(box, (tuple, list)) or len(box) != 4 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in box) or
                box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Invalid model-space box")
        left, top, right, bottom = (
            (box[0] - self.pad_x) / self.scale_x,
            (box[1] - self.pad_y) / self.scale_y,
            (box[2] - self.pad_x) / self.scale_x,
            (box[3] - self.pad_y) / self.scale_y,
        )
        if clamp:
            left, right = max(0, min(left, self.source_width)), max(0, min(right, self.source_width))
            top, bottom = max(0, min(top, self.source_height)), max(0, min(bottom, self.source_height))
        return left, top, right, bottom
