import hashlib

import pytest

from cardviper_ml.detector_config import DetectorConfig, training_plan
from cardviper_ml.manifest import ImageRecord
from cardviper_ml.splits import write_splits


def row(name, *, held=False):
    return ImageRecord(source_id="source", scene_id=name, session_id=name,
                       image_path=f"{name}.png", width=20, height=20,
                       image_sha256=hashlib.sha256(name.encode()).hexdigest(),
                       annotation_path=f"{name}.json", objects=(), held_out=held)


def test_one_class_320_config_and_split_use(tmp_path):
    config = DetectorConfig(seed=7)
    assert config.class_names == ("CARD",)
    assert config.input_size == (320, 320)
    splits = {"train": [row("train")], "val": [row("val")],
              "test": [row("test")], "holdout": [row("held", held=True)]}
    write_splits(tmp_path / "splits", splits)
    plan = training_plan(tmp_path / "splits", config)
    assert plan["fit_split"] == "train"
    assert plan["selection_split"] == "val"
    assert plan["reserved_splits"] == ["test", "holdout"]
    assert plan["seed"] == 7
    assert plan["class_names"] == ["CARD"]
    assert plan["split_counts"] == {name: 1 for name in splits}


def test_invalid_class_or_seed_and_leaky_splits_fail(tmp_path):
    with pytest.raises(ValueError, match="CARD"):
        DetectorConfig(class_names=("AC", "BACK"))
    with pytest.raises(ValueError, match="seed"):
        DetectorConfig(seed=True)
    splits = {"train": [row("same")], "val": [row("same")],
              "test": [row("test")], "holdout": []}
    with pytest.raises(ValueError):
        write_splits(tmp_path / "leaky", splits)
