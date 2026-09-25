"""Canonical Android-compatible labels; explicit future classifier index order."""

import argparse
from dataclasses import dataclass
from pathlib import Path

# Deliberately literal: enum/dict iteration never determines model output indices.
LABELS = (
    "AC", "AD", "AH", "AS", "2C", "2D", "2H", "2S",
    "3C", "3D", "3H", "3S", "4C", "4D", "4H", "4S",
    "5C", "5D", "5H", "5S", "6C", "6D", "6H", "6S",
    "7C", "7D", "7H", "7S", "8C", "8D", "8H", "8S",
    "9C", "9D", "9H", "9S", "10C", "10D", "10H", "10S",
    "JC", "JD", "JH", "JS", "QC", "QD", "QH", "QS",
    "KC", "KD", "KH", "KS", "BACK",
)


@dataclass(frozen=True)
class Face:
    rank: str
    suit: str

    def __post_init__(self):
        if self.rank not in ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K") or self.suit not in ("C", "D", "H", "S"):
            raise ValueError("Invalid face rank/suit")


@dataclass(frozen=True)
class Back:
    """Face-down card, intentionally without rank or suit."""


def decode(label: str) -> Face | Back:
    if not isinstance(label, str) or label not in LABELS:
        raise ValueError(f"Invalid canonical card label: {label!r}")
    return Back() if label == "BACK" else Face(label[:-1], label[-1])


def encode(identity: Face | Back) -> str:
    if isinstance(identity, Back):
        return "BACK"
    if isinstance(identity, Face):
        return identity.rank + identity.suit
    raise ValueError("Expected Face or Back")


def write_labels(path: Path) -> None:
    Path(path).write_text("\n".join(LABELS) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    write_labels(parser.parse_args().output)


if __name__ == "__main__":
    main()
