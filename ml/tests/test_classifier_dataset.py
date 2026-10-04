import json

from PIL import Image
import pytest

from cardviper_ml.classifier_dataset import load_crop_split, seeded_batches, verify_crop_provenance


def test_manifest_controls_label_and_seeded_batch_order(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    for i in range(4):
        Image.new("RGB", (8, 8), (i, 0, 0)).save(root / f"{i}.png")
    rows = [dict(schema_version=1, source_id="owned", scene_id=f"s{i}", session_id=None,
                 held_out=False, source_image=f"source/{i}.png", source_image_sha256=f"{i:064x}",
                 source_annotation=f"ann/{i}.txt", annotation_id="1", source_bbox=[0, 0, 8, 8],
                 label="AC", crop_bbox=[0, 0, 8, 8], padding=0, crop_path=f"{i}.png") for i in range(4)]
    (root / "crops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    samples = load_crop_split(root, "train")
    assert [s.label_index for s in samples] == [0] * 4
    assert [[s.crop_path.name for s in batch] for batch in seeded_batches(samples, 2, 42)] == [
        [s.crop_path.name for s in batch] for batch in seeded_batches(samples, 2, 42)]
    rows[0]["label_index"] = 52
    (root / "crops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="index"):
        load_crop_split(root, "train")


def test_missing_crop_fails(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    (root / "crops.jsonl").write_text(json.dumps({"label": "BACK", "crop_path": "missing.png"}) + "\n")
    with pytest.raises(ValueError):
        load_crop_split(root, "train")


def test_group_reserved_holdout_accepts_member_without_individual_held_out_flag(tmp_path):
    root = tmp_path / "holdout"
    root.mkdir()
    Image.new("RGB", (8, 8), "blue").save(root / "card.png")
    row = dict(schema_version=1, source_id="owned", scene_id="shared-scene", session_id=None,
               held_out=False, source_image="source/card.png", source_image_sha256="a" * 64,
               source_annotation="ann/card.txt", annotation_id="1", source_bbox=[0, 0, 8, 8],
               label="BACK", crop_bbox=[0, 0, 8, 8], padding=0, crop_path="card.png")
    (root / "crops.jsonl").write_text(json.dumps(row) + "\n")
    assert load_crop_split(root, "holdout")[0].label == "BACK"


def test_crop_provenance_checks_geometry_filename_and_actual_pixels(tmp_path):
    import hashlib
    from cardviper_ml.build_classifier_crops import build_crops
    from cardviper_ml.manifest import Annotation, ImageRecord
    root = tmp_path / "source"
    root.mkdir()
    source = root / "a.png"
    Image.new("RGB", (12, 12), "red").save(source)
    row = ImageRecord(source_id="owned", scene_id="scene-1", image_path="a.png",
                      width=12, height=12, image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      annotation_path="labels/a.txt", objects=(Annotation("1", "AC", (2, 2, 8, 8)),))
    crops = tmp_path / "train"
    build_crops([row], {"owned": root}, crops, padding=.1)
    assert len(verify_crop_provenance(crops, [row], {"owned": root}, "train")) == 1
    path = next(crops.glob("*.png"))
    Image.new("RGB", (8, 8), "blue").save(path)
    with pytest.raises(ValueError, match="pixels"):
        verify_crop_provenance(crops, [row], {"owned": root}, "train")
