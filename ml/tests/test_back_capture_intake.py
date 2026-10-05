import hashlib
import json

from PIL import Image
import pytest

from cardviper_ml.back_capture_intake import intake, validate


def photo(root, name, color):
    path = root / "inbox" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 16), color).save(path)
    return path


def test_absent_source_stays_not_ready(tmp_path):
    result = validate(tmp_path / "absent")
    assert result["status"] == "NOT READY"
    assert result["blocker"] == "BACK_SOURCE_MISSING"
    assert result["image_files"] == 0


def test_intake_requires_explicit_metadata_and_preserves_entries(tmp_path):
    first = photo(tmp_path, "session-1/a.png", "red")
    report = intake(tmp_path)
    assert report["status"] == "NOT READY"
    manifest = tmp_path / "captures.json"
    entry = json.loads(manifest.read_text())["captures"][0]
    assert entry["image_path"] == "inbox/session-1/a.png"
    assert entry["image_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert entry["width"] == 24 and entry["height"] == 16
    assert entry["capture_date"] == entry["device"] == entry["session_id"] == ""
    assert entry["full_image_back"] is False
    original = manifest.read_bytes()
    assert validate(tmp_path)["incomplete_entries"] == 1
    assert intake(tmp_path) == report
    assert manifest.read_bytes() == original
    photo(tmp_path, "session-2/b.png", "blue")
    intake(tmp_path)
    entries = json.loads(manifest.read_text())["captures"]
    assert entries[0] == entry
    assert len(entries) == 2


def test_explicit_batch_and_per_image_metadata_without_approval(tmp_path):
    photo(tmp_path, "s1/a.png", "red")
    photo(tmp_path, "s2/b.png", "blue")
    metadata = {
        "common": {"owner": "Photographer", "rights": {"classifier": True, "detector": False,
                    "evidence": "Consent record 1"}, "device": "Camera 1",
                   "capture_date": "2026-10-04", "deck_design": "blue-pattern",
                   "annotation_provenance": "Human crop review", "capture_mode": "tight-back-crop"},
        "images": {
            "inbox/s1/a.png": {"session_id": "s1", "scene_id": "s1-a", "full_image_back": True},
            "inbox/s2/b.png": {"session_id": "s2", "scene_id": "s2-b", "full_image_back": True,
                               "deck_design": "red-pattern"},
        },
    }
    intake(tmp_path, metadata)
    result = validate(tmp_path)
    assert result["complete_entries"] == 2
    assert result["incomplete_entries"] == 0
    assert result["declared_sessions"] == 2
    assert result["independent_sessions"] == 2  # structural groups, not human confirmation
    assert result["session_independence_confirmed"] is False
    assert result["deck_designs"] == 2
    assert result["status"] == "NOT READY"  # existing audit still needs 3 groups and human review
    assert json.loads((tmp_path / "captures.json").read_text())["review"]["approved"] is False


def test_duplicate_and_hash_mismatch_reported_without_rewriting(tmp_path):
    first = photo(tmp_path, "a/one.png", "red")
    second = photo(tmp_path, "b/two.png", "red")
    intake(tmp_path)
    report = validate(tmp_path)
    assert report["exact_duplicates"] == [["inbox/a/one.png", "inbox/b/two.png"]]
    Image.new("RGB", (24, 16), "green").save(first)
    assert "inbox/a/one.png" in validate(tmp_path)["hash_mismatches"]
    before = (tmp_path / "captures.json").read_bytes()
    with pytest.raises(ValueError, match="changed since intake"):
        intake(tmp_path)
    assert (tmp_path / "captures.json").read_bytes() == before
    second.unlink()
    assert "inbox/b/two.png" in validate(tmp_path)["missing_files"]


def test_metadata_conflict_does_not_overwrite(tmp_path):
    photo(tmp_path, "s/a.png", "red")
    intake(tmp_path, {"images": {"inbox/s/a.png": {"device": "Camera 1"}}})
    before = (tmp_path / "captures.json").read_bytes()
    with pytest.raises(ValueError, match="conflict"):
        intake(tmp_path, {"images": {"inbox/s/a.png": {"device": "Camera 2"}}})
    assert (tmp_path / "captures.json").read_bytes() == before


def test_existing_contract_entry_without_dimensions_is_preserved(tmp_path):
    path = photo(tmp_path, "s/a.png", "red")
    entry = {"image_path": "inbox/s/a.png", "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "capture_date": "2026-10-04", "device": "Camera 1", "session_id": "s",
             "scene_id": "s-a", "deck_design": "pattern", "annotation_provenance": "Human",
             "capture_mode": "tight-back-crop", "full_image_back": True}
    data = {"schema_version": 1, "source_id": "cardviper-back", "owner": "Photographer",
            "rights": {"classifier": True, "detector": False, "evidence": "Consent"},
            "captures": [entry], "review": {"approved": False}}
    manifest = tmp_path / "captures.json"
    manifest.write_text(json.dumps(data))
    intake(tmp_path)
    assert json.loads(manifest.read_text())["captures"][0] == entry


def test_global_missing_metadata_is_reported(tmp_path):
    photo(tmp_path, "s/a.png", "red")
    intake(tmp_path, {"common": {"capture_mode": "annotated-scene", "device": "Camera"},
                      "images": {"inbox/s/a.png": {"boxes": [{"annotation_id": "back-1",
                                 "label": "BACK", "bbox": [1, 1, 20, 12]}]}}})
    report = validate(tmp_path)
    assert "owner" in report["missing_fields"]["source"]
    assert "rights.evidence" in report["missing_fields"]["source"]
    assert report["annotation_modes"] == {"annotated-scene": 1}
    assert "capture_date" in report["missing_fields"]["inbox/s/a.png"]
