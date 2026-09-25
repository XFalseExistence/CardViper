"""Library-independent, deterministic B2 classifier configuration contract."""

from dataclasses import dataclass
from pathlib import Path

from .labels import LABELS


@dataclass(frozen=True)
class ClassifierDataPaths:
    train: Path
    validation: Path
    test: Path
    holdout: Path

    def __post_init__(self):
        values = tuple(Path(value) for value in (
            self.train, self.validation, self.test, self.holdout
        ))
        if len({value.resolve(strict=False) for value in values}) != 4:
            raise ValueError("Classifier train, validation, test and holdout paths must be distinct")
        for field, value in zip(("train", "validation", "test", "holdout"), values):
            object.__setattr__(self, field, value)

    @classmethod
    def from_split_directory(cls, directory):
        directory = Path(directory)
        return cls(
            train=directory / "train.jsonl",
            validation=directory / "val.jsonl",
            test=directory / "test.jsonl",
            holdout=directory / "holdout.jsonl",
        )


@dataclass(frozen=True)
class ClassifierTrainingConfig:
    labels: tuple[str, ...]
    declared_output_width: int
    seed: int
    data: ClassifierDataPaths

    def __post_init__(self):
        labels = tuple(self.labels)
        if labels != LABELS:
            raise ValueError("Classifier labels must equal the canonical explicit 53-label order")
        if type(self.declared_output_width) is not int or self.declared_output_width != 53:
            raise ValueError("Classifier output width must be exactly 53")
        if type(self.seed) is not int:
            raise ValueError("A deterministic integer seed must be explicit")
        if not isinstance(self.data, ClassifierDataPaths):
            raise ValueError("Classifier data paths must be validated")
        object.__setattr__(self, "labels", labels)

    @property
    def output_width(self):
        return len(self.labels)

    @property
    def output_index_to_label(self):
        return tuple(enumerate(self.labels))

    @property
    def training_inputs(self):
        """Only inputs allowed during fitting/checkpoint selection."""
        return self.data.train, self.data.validation
