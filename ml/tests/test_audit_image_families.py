from cardviper_ml.audit_image_families import audit_families, filename_family
from cardviper_ml.audit_image_families import collect_rows
from PIL import Image


def test_roboflow_variants_share_candidate_family():
    assert filename_family("train/images/card_jpg.rf.abc.jpg") == filename_family("valid/images/card_jpg.rf.def.jpg")
    rows = [
        {"path": "train/images/card_jpg.rf.abc.jpg", "sha256": "a" * 64, "width": 32, "height": 32, "labels": ["AC"]},
        {"path": "valid/images/card_jpg.rf.def.jpg", "sha256": "b" * 64, "width": 32, "height": 32, "labels": ["AC"]},
    ]
    report = audit_families(rows)
    assert len(report["probable_augmentation_families"]) == 1
    assert report["grouping_ready"] is False


def test_exact_duplicates_detected_across_splits():
    rows = [{"path": "train/images/a.png", "sha256": "a" * 64, "width": 1, "height": 1, "labels": []},
            {"path": "test/images/b.png", "sha256": "a" * 64, "width": 1, "height": 1, "labels": []}]
    assert len(audit_families(rows)["exact_duplicates"]) == 1


def test_nested_export_images_are_not_omitted(tmp_path):
    image = tmp_path / "train/images/nested/a.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(image)
    label = tmp_path / "train/labels/nested/a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 .5 .5 .5 .5\n")
    rows = collect_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["annotation_exists"] is True
