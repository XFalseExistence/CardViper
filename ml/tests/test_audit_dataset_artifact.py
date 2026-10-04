import json
from pathlib import Path

from PIL import Image

from cardviper_ml.labels import LABELS
from cardviper_ml.audit_dataset_artifact import audit_artifact


def test_exact_52_class_export_is_valid_but_not_training_ready(tmp_path):
    (tmp_path / "train/images").mkdir(parents=True)
    (tmp_path / "train/labels").mkdir()
    (tmp_path / "data.yaml").write_text("roboflow:\n  workspace: joshuas-workspace\n  project: playing-cards-9gfac\n  version: 2\nnames:\n" + "".join(
        f"  {i}: {label}\n" for i, label in enumerate(LABELS[:-1])))
    Image.new("RGB", (16, 16), "red").save(tmp_path / "train/images/a.png")
    (tmp_path / "train/labels/a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    report = audit_artifact(tmp_path, expected_source_id="playing-cards-seed")
    assert report["artifact_status"] == "SOURCE_ARTIFACT_VALID"
    assert report["training_status"] == "NOT READY"
    assert report["class_index_to_label"]["0"] == "AC"
    assert report["annotation_count"] == 1
    (tmp_path / "train/labels/a.txt").write_text("52 0.5 0.5 0.5 0.5\n")
    assert audit_artifact(tmp_path, expected_source_id="playing-cards-seed")["artifact_status"] == "INVALID"


def test_wrong_source_id_fails_closed(tmp_path):
    assert audit_artifact(tmp_path, expected_source_id="other")["artifact_status"] == "INVALID"


def test_wrong_project_metadata_fails_closed(tmp_path):
    (tmp_path / "data.yaml").write_text("roboflow:\n  workspace: other\n  project: playing-cards-9gfac\n  version: 2\nnames: []\n")
    assert audit_artifact(tmp_path)["artifact_status"] == "INVALID"


def test_literal_back_cannot_be_aliased_into_face_export(tmp_path):
    (tmp_path / "data.yaml").write_text("roboflow:\n  workspace: joshuas-workspace\n  project: playing-cards-9gfac\n  version: 2\nnames:\n" + "".join(
        f"  {i}: {label}\n" for i, label in enumerate((*LABELS[1:-1], "BACK"))))
    report = audit_artifact(tmp_path, class_map={"BACK": "AC"})
    assert report["artifact_status"] == "INVALID"
    assert "literal BACK" in report["errors"][0]


def test_mixed_rows_are_audited_without_changing_source(tmp_path):
    (tmp_path / "train/images").mkdir(parents=True)
    (tmp_path / "train/labels").mkdir()
    (tmp_path / "data.yaml").write_text("roboflow:\n  workspace: joshuas-workspace\n  project: playing-cards-9gfac\n  version: 2\nnames:\n" + "".join(
        f"  {i}: {label}\n" for i, label in enumerate(LABELS[:-1])))
    Image.new("RGB", (16, 16), "red").save(tmp_path / "train/images/a.png")
    label = tmp_path / "train/labels/a.txt"
    label.write_text("0 .5 .5 .4 .4\n1 .1 .1 .9 .1 .5 .9\n")
    before = label.read_bytes()
    report = audit_artifact(tmp_path)
    assert report["artifact_status"] == "SOURCE_ARTIFACT_VALID"
    assert (report["bbox_row_count"], report["polygon_row_count"],
            report["malformed_row_count"], report["normalized_object_count"]) == (1, 1, 0, 2)
    assert report["polygon_conversions"][0]["label_path"] == "train/labels/a.txt"
    assert report["polygon_conversions"][0]["row"] == 2
    assert report["polygon_conversions"][0]["point_count"] == 3
    assert label.read_bytes() == before
