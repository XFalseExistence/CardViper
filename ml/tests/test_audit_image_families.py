from cardviper_ml.audit_image_families import audit_families, filename_family, video_recording_family
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


def test_exported_filename_lineage_and_recording_keys():
    one = "train/images/VID_20230227_221709_mp4-0_jpg.rf.2d47a4aec28c20b7204aa7e9050b5b4d.jpg"
    two = "valid/images/VID_20230227_221709_mp4-0_jpg.rf.db36d6892c6a2cc6faa5798aaa492aee.jpg"
    other_frame = "test/images/VID_20230227_221709_mp4-19_jpg.rf.27065ecb9d70197211f915facc5127a0.jpg"
    assert filename_family(one) == filename_family(two)
    assert filename_family(one) != filename_family(other_frame)
    assert video_recording_family(one) == video_recording_family(other_frame)
    assert video_recording_family("train/images/frame_0001_png.rf.c239ca335d409e9c490f1e32d279410e.jpg") is None


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
