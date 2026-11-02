"""Pure boundaries for the one-class Session A detector trainer."""

import hashlib
import os

from cardviper_ml.manifest import Annotation, ImageRecord


def _row() -> ImageRecord:
    return ImageRecord(
        source_id="playing-cards-seed",
        scene_id="session-a",
        session_id="session-a",
        image_path="train/images/table.png",
        width=100,
        height=80,
        image_sha256=hashlib.sha256(b"table").hexdigest(),
        annotation_path="train/labels/table.txt",
        objects=(
            Annotation("1", "AC", (0, 0, 25, 40)),
            Annotation("2", "KS", (40, 20, 100, 80)),
        ),
    )


def test_detector_example_collapses_face_labels_to_card_class_zero():
    """A regression that leaks face identities into detector classes must fail."""
    from cardviper_ml.train_session_a_demo_detector import detector_example

    example = detector_example(_row())

    assert example["class_ids"] == [0, 0]
    assert example["class_names"] == ["CARD", "CARD"]
    assert example["xmins"] == [0.0, 0.4]
    assert example["ymaxs"] == [0.5, 1.0]


def test_pinned_fpn_lite_settings_are_one_class_320_only():
    """A trainer that silently selects a different detector surface must fail."""
    from cardviper_ml.train_session_a_demo_detector import detector_pipeline_settings

    settings = detector_pipeline_settings(batch_size=4, train_steps=50)

    assert settings["config_relative_path"] == (
        "research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config"
    )
    assert settings["class_names"] == ["CARD"]
    assert settings["num_classes"] == 1
    assert settings["input_size"] == [320, 320]
    assert settings["batch_size"] == 4
    assert settings["train_steps"] == 50


def test_odapi_subprocess_gets_research_and_slim_import_paths(tmp_path):
    """A trainer subprocess unable to import its pinned API must fail this test."""
    from cardviper_ml.train_session_a_demo_detector import odapi_environment

    environment = odapi_environment(tmp_path, {"PYTHONPATH": "existing"})

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path / "research"), str(tmp_path / "research" / "slim"), "existing",
    ]
