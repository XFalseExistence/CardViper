import hashlib
import json

from PIL import Image
import pytest

from cardviper_ml.import_detector_negatives import import_negatives


def fixture(root, *, confirmed=True):
    image = root / "session-1" / "table.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (20, 10), "green").save(image)
    data = {"schema_version": 1, "source_id": "cardviper-detector-negatives",
            "owner": "Photographer", "rights": {"detector": True, "evidence": "Consent record"},
            "scenes": [{"scene_id": "table-1", "session_id": "session-1",
                        "capture_date": "2026-10-04", "device": "Camera 1",
                        "image_path": "session-1/table.png",
                        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "annotation_provenance": "Human frame inspection",
                        "no_card_confirmed": confirmed, "held_out": True}]}
    (root / "scenes.json").write_text(json.dumps(data))
    return data


def test_reviewed_negative_imports_as_empty_objects_with_provenance(tmp_path):
    fixture(tmp_path)
    row = import_negatives(tmp_path)[0]
    assert row.objects == ()
    assert row.source_id == "cardviper-detector-negatives"
    assert row.scene_id == "table-1" and row.session_id == "session-1"
    assert row.held_out is True and row.annotation_path == "scenes.json"
    assert (row.width, row.height) == (20, 10)


def test_unconfirmed_scene_or_wrong_hash_cannot_become_negative(tmp_path):
    data = fixture(tmp_path, confirmed=False)
    with pytest.raises(ValueError, match="no_card_confirmed"):
        import_negatives(tmp_path)
    data["scenes"][0]["no_card_confirmed"] = True
    data["scenes"][0]["image_sha256"] = "0" * 64
    (tmp_path / "scenes.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        import_negatives(tmp_path)


def test_rights_and_annotation_provenance_required(tmp_path):
    data = fixture(tmp_path)
    data["rights"]["detector"] = False
    (tmp_path / "scenes.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="detector rights"):
        import_negatives(tmp_path)
    data["rights"]["detector"] = True
    data["scenes"][0]["annotation_provenance"] = ""
    (tmp_path / "scenes.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="annotation_provenance"):
        import_negatives(tmp_path)
