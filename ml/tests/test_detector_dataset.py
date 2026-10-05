import hashlib
import json

import pytest

from cardviper_ml.manifest import Annotation, ImageRecord, read_manifest, write_manifest
from cardviper_ml.build_detector_records import build_detector_records, write_detector_records


def record(path, objects, *, scene="scene", session="session", held=False):
    return ImageRecord(source_id="approved-source", scene_id=scene, session_id=session,
                       image_path=path, width=100, height=80,
                       image_sha256=hashlib.sha256(path.encode()).hexdigest(),
                       annotation_path="annotations.json", objects=objects, held_out=held)


def test_face_back_and_multiple_cards_collapse_to_one_detector_class(tmp_path):
    objects = (Annotation("a", "AC", (0, 0, 20, 30)),
               Annotation("b", "BACK", (30, 10, 60, 70)),
               Annotation("c", "2S", (70, 0, 100, 80)))
    row = record("image.png", objects, held=True)
    view = build_detector_records([row])[0]
    assert view["source_id"] == row.source_id
    assert view["scene_id"] == row.scene_id and view["session_id"] == row.session_id
    assert view["image_path"] == row.image_path and view["image_sha256"] == row.image_sha256
    assert view["held_out"] is True
    assert view["group_keys"] == [["approved-source", "scene", "scene"],
                                  ["approved-source", "session", "session"]]
    assert len(view["objects"]) == 3
    assert {obj["class_id"] for obj in view["objects"]} == {0}
    assert {obj["class_name"] for obj in view["objects"]} == {"CARD"}
    assert [obj["bbox"] for obj in view["objects"]] == [[0, 0, 20, 30],
                                                        [30, 10, 60, 70], [70, 0, 100, 80]]
    assert all("label" not in obj for obj in view["objects"])
    output = tmp_path / "detector.jsonl"
    write_detector_records(output, [row])
    assert json.loads(output.read_text().strip()) == view


def test_explicit_no_card_scene_is_preserved_and_missing_annotations_fail(tmp_path):
    negative = record("empty.png", (), scene="no-card", session="other")
    assert build_detector_records([negative])[0]["objects"] == []
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(json.dumps({"source_id": "x", "scene_id": "s", "image_path": "a.png",
                                    "width": 10, "height": 10, "image_sha256": "0" * 64,
                                    "annotation_path": "a.json"}) + "\n")
    with pytest.raises(ValueError, match="objects"):
        read_manifest(manifest)


def test_invalid_boxes_and_group_boundary_rejected_or_preserved():
    with pytest.raises(ValueError, match="outside source image"):
        record("bad.png", (Annotation("a", "AC", (0, 0, 101, 20)),))
    rows = [record("train.png", (), scene="one", session="s1"),
            record("held.png", (), scene="two", session="s2", held=True)]
    views = build_detector_records(rows)
    assert views[0]["group_keys"] != views[1]["group_keys"]
    assert views[0]["held_out"] is False and views[1]["held_out"] is True
