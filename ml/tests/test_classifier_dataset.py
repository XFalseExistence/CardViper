import json

from PIL import Image
import pytest

from cardviper_ml.classifier_dataset import load_crop_split, seeded_batches


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
