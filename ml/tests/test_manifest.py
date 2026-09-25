from dataclasses import replace
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest


def record(**changes):
    from cardviper_ml.manifest import Annotation, ImageRecord
    values = dict(source_id="seed", scene_id="scene-1", session_id="session-1",
                  image_path="images/a.png", width=100, height=80,
                  image_sha256=hashlib.sha256(b"image-a").hexdigest(),
                  annotation_path="labels/a.txt",
                  objects=(Annotation("line-1", "7C", (10, 20, 40, 60)),))
    values.update(changes)
    return ImageRecord(**values)


def test_manifest_roundtrip_retains_provenance_and_negative_images(tmp_path):
    from cardviper_ml.manifest import read_manifest, write_manifest
    rows = [record(), record(image_path="images/negative.png", objects=(),
                             image_sha256=hashlib.sha256(b"negative").hexdigest())]
    target = tmp_path / "data.jsonl"
    write_manifest(target, rows)
    assert read_manifest(target) == rows
    raw = json.loads(target.read_text().splitlines()[0])
    assert raw["objects"][0]["detector_class"] == "CARD"
    assert raw["objects"][0]["bbox"] == [10, 20, 40, 60]
    assert raw["annotation_path"] == "labels/a.txt"


@pytest.mark.parametrize("change", [
    {"source_id": ""}, {"scene_id": " "}, {"scene_id": None},
    {"width": 0}, {"height": -1}, {"width": 3.5}, {"width": True},
    {"image_path": "../escape.jpg"}, {"image_path": "/private/photo.jpg"},
    {"image_sha256": "unknown"}, {"session_id": ""},
])
def test_invalid_record_fails_closed(change):
    with pytest.raises(ValueError):
        record(**change)


@pytest.mark.parametrize("box", [
    (0, 0, 0, 1), (9, 2, 1, 8), (-1, 0, 4, 5), (0, 0, 101, 50),
    (100, 20, 110, 30), (0, 0, float("nan"), 5), (0, 0, 4, float("inf")),
    (1, 2, 3),
])
def test_invalid_box_rejected(box):
    from cardviper_ml.manifest import Annotation
    with pytest.raises(ValueError):
        record(objects=(Annotation("1", "AC", box),))


def test_unknown_label_and_non_card_detector_class_rejected():
    from cardviper_ml.manifest import Annotation
    with pytest.raises(ValueError):
        Annotation("1", "JOKER", (0, 0, 1, 1))
    with pytest.raises(ValueError):
        Annotation("1", "AC", (0, 0, 1, 1), detector_class="AC")


def test_duplicate_annotation_ids_and_boxes_rejected():
    from cardviper_ml.manifest import Annotation
    obj = record().objects[0]
    for other in (obj, replace(obj, annotation_id="2")):
        with pytest.raises(ValueError, match="Duplicate"):
            record(objects=(obj, other))


def export_fixture(tmp_path, text):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.new("RGB", (100, 80), "green").save(tmp_path / "images/a.png")
    (tmp_path / "labels/a.txt").write_text(text)
    return {"images/a.png": {"scene_id": "scene-1", "session_id": "capture-1"}}


def test_local_yolo_import_preserves_identity_collapses_detector_and_keeps_negatives(tmp_path):
    from cardviper_ml.import_roboflow import import_yolo
    groups = export_fixture(tmp_path, "0 .25 .5 .2 .5\n1 .5 .5 .2 .5\n2 .75 .5 .2 .5\n3 .5 .2 .1 .1\n4 .9 .9 .1 .1\n")
    Image.new("RGB", (100, 80), "black").save(tmp_path / "images/no-cards.png")
    (tmp_path / "labels/no-cards.txt").write_text("")
    groups["images/no-cards.png"] = {"scene_id": "negative"}
    rows = import_yolo(tmp_path, "seed", ["7C", "KH", "2D", "10S", "BACK"], groups)
    assert len(rows) == 2
    assert [a.label for a in rows[0].objects] == ["7C", "KH", "2D", "10S", "BACK"]
    assert {a.detector_class for a in rows[0].objects} == {"CARD"}
    assert rows[0].objects[0].bbox == pytest.approx((15, 20, 35, 60))
    assert rows[0].annotation_path == "labels/a.txt"
    assert rows[0].image_sha256 == hashlib.sha256((tmp_path / "images/a.png").read_bytes()).hexdigest()
    assert rows[1].objects == ()


@pytest.mark.parametrize("line", ["5 .5 .5 .1 .1", "-1 .5 .5 .1 .1", "0.0 .5 .5 .1 .1",
    "0 1.1 .5 .1 .1", "0 0 .5 .2 .1", "0 .5 .5 0 .1", "0 nan .5 .1 .1",
    "0 .5 .5 .1 inf", "0 .5 .5 .1", "0 .5 .5 .1 .1 .9"])
def test_import_invalid_yolo_annotation_fails(tmp_path, line):
    from cardviper_ml.import_roboflow import import_yolo
    groups = export_fixture(tmp_path, line)
    with pytest.raises(ValueError):
        import_yolo(tmp_path, "seed", ["AC"], groups)


def test_unknown_class_requires_explicit_alias_mapping(tmp_path):
    from cardviper_ml.import_roboflow import import_yolo
    groups = export_fixture(tmp_path, "0 .5 .5 .2 .2")
    for name in ("JOKER", "ac"):
        with pytest.raises(ValueError):
            import_yolo(tmp_path, "seed", [name], groups)
    assert import_yolo(tmp_path, "seed", ["ac"], groups, class_map={"ac": "AC"})[0].objects[0].label == "AC"


def test_missing_groups_or_annotation_file_not_assumed_negative(tmp_path):
    from cardviper_ml.import_roboflow import import_yolo
    groups = export_fixture(tmp_path, "")
    with pytest.raises(ValueError, match="group"):
        import_yolo(tmp_path, "seed", ["AC"], {})
    (tmp_path / "labels/a.txt").unlink()
    with pytest.raises(ValueError, match="annotation"):
        import_yolo(tmp_path, "seed", ["AC"], groups)


def test_source_metadata_blocks_unverified_training(tmp_path):
    from cardviper_ml.manifest import read_sources, require_training_permission
    sources = read_sources(Path(__file__).parents[1] / "datasets/sources.json")
    seed = sources["playing-cards-seed"]
    assert seed["claimed_license"] == "CC-BY-4.0"
    assert seed["url"] == "https://universe.roboflow.com/joshuas-workspace/playing-cards-9gfac"
    assert seed["version"] == "2"
    assert seed["license"] == "CC-BY-4.0"
    assert seed["attribution"] == "Playing Cards Dataset by Joshuas Workspace, Roboflow Universe (2024)"
    assert seed["source_page_verified"] is True
    assert seed["artifact_verified"] is False
    assert seed["export"]["format"] == "YOLOv8"
    assert seed["export"]["download_requires_login"] is True
    assert seed["export"]["class_index_order_verified"] is False
    assert seed["grouping_verified"] is False
    assert not seed["verified"]
    for purpose in ("detector", "classifier"):
        with pytest.raises(ValueError, match="verified"):
            require_training_permission(sources, "playing-cards-seed", purpose)
    source = dict(seed, source_id="owned", name="Owned synthetic", url="https://example.org/owned",
                  license="CC0-1.0", attribution="Fixture", verified=True,
                  permitted_use={"detector": True, "classifier": False})
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"schema_version": 1, "sources": [source]}))
    verified = read_sources(path)
    require_training_permission(verified, "owned", "detector")
    with pytest.raises(ValueError):
        require_training_permission(verified, "owned", "classifier")
    source["license"] = None
    path.write_text(json.dumps({"schema_version": 1, "sources": [source]}))
    with pytest.raises(ValueError):
        read_sources(path)
