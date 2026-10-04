import hashlib
import json

from PIL import Image
import pytest

from cardviper_ml.prepare_back_capture_manifest import prepare_manifest
from cardviper_ml.import_back_captures import import_back_source


def test_draft_hashes_images_but_never_grants_rights_or_full_image_annotation(tmp_path):
    root = tmp_path / "cardviper-back"
    image = root / "session-001" / "back.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (20, 12), "blue").save(image)
    original = image.read_bytes()
    report = prepare_manifest(root, device="Pixel 7", capture_mode="tight-back-crop")
    assert report["status"] == "DRAFT"
    data = json.loads((root / "captures.json").read_text())
    assert data["rights"]["classifier"] is False
    assert data["review"]["approved"] is False
    row = data["captures"][0]
    assert row["image_path"] == "session-001/back.jpg"
    assert row["image_sha256"] == hashlib.sha256(original).hexdigest()
    assert row["width"] == 20 and row["height"] == 12
    assert row["session_id"] == "session-001"
    assert row["device"] == "Pixel 7"
    assert row["full_image_back"] is False
    assert image.read_bytes() == original
    with pytest.raises(ValueError):
        import_back_source(root)


def test_existing_manifest_is_never_overwritten(tmp_path):
    root = tmp_path / "back"
    image = root / "session-001" / "a.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(image)
    (root / "captures.json").write_text("owned")
    with pytest.raises(FileExistsError):
        prepare_manifest(root, device="camera", capture_mode="annotated-scene")
    assert (root / "captures.json").read_text() == "owned"


def test_annotated_scene_draft_has_no_invented_boxes(tmp_path):
    root = tmp_path / "back"
    image = root / "session-001" / "a.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(image)
    prepare_manifest(root, device="camera", capture_mode="annotated-scene")
    row = json.loads((root / "captures.json").read_text())["captures"][0]
    assert row["boxes"] == []
    assert "full_image_back" not in row
