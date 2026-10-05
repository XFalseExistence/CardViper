"""One-class detector data-use policy; this module does not train a model."""

from dataclasses import dataclass
from pathlib import Path

from .splits import read_splits


@dataclass(frozen=True)
class DetectorConfig:
    class_names: tuple[str, ...] = ("CARD",)
    input_size: tuple[int, int] = (320, 320)
    seed: int = 42

    def __post_init__(self):
        if self.class_names != ("CARD",):
            raise ValueError("Detector semantic classes must be exactly CARD")
        if self.input_size != (320, 320):
            raise ValueError("Preferred detector input must be 320x320")
        if type(self.seed) is not int:
            raise ValueError("Detector seed must be an explicit integer")


def training_plan(splits_directory: Path, config: DetectorConfig):
    """Validate existing grouped splits and name the only permitted fit/selection roles."""
    if not isinstance(config, DetectorConfig):
        raise ValueError("Expected DetectorConfig")
    splits = read_splits(splits_directory)
    return {
        "class_names": list(config.class_names),
        "input_size": list(config.input_size),
        "seed": config.seed,
        "fit_split": "train",
        "selection_split": "val",
        "reserved_splits": ["test", "holdout"],
        "split_counts": {name: len(rows) for name, rows in splits.items()},
    }
