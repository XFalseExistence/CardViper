from cardviper_ml.eval_classifier import evaluate_scores
from cardviper_ml.labels import LABELS
import hashlib
from dataclasses import replace

import pytest

from cardviper_ml.classifier_dataset import CropSample
from cardviper_ml.eval_classifier import validate_test_provenance
from cardviper_ml.manifest import Annotation, ImageRecord, write_manifest


def _scores(label, confidence=.9):
    result = [(1 - confidence) / 52] * 53
    result[LABELS.index(label)] = confidence
    return result


def test_exact_top1_back_and_red_black_twos():
    report = evaluate_scores(["2C", "2D", "BACK"],
                             [_scores("2C"), _scores("2H"), _scores("BACK")], split="test")
    assert report["top1_accuracy"] == 2 / 3
    assert report["back"]["recall"] == 1
    assert report["back"]["precision"] == 1
    assert report["twos"]["red_vs_black_confusions"] == 0
    assert report["twos"]["confusion_matrix"]["2D"]["2H"] == 1
    assert report["confusion_matrix"]["2D"]["2H"] == 1


def test_holdout_evaluation_is_blocked():
    with pytest.raises(ValueError, match="holdout"):
        evaluate_scores(["AC"], [_scores("AC")], split="holdout")


def test_test_evaluation_rejects_crop_or_split_substitution(tmp_path):
    model = tmp_path / "chosen.keras"
    model.write_bytes(b"chosen-checkpoint")
    split = tmp_path / "test.jsonl"
    row = ImageRecord(source_id="owned", scene_id="scene-1", image_path="images/a.png",
                      width=10, height=10, image_sha256="a" * 64,
                      annotation_path="labels/a.txt", objects=(Annotation("1", "AC", (0, 0, 5, 5)),))
    write_manifest(split, [row])
    crop_manifest = tmp_path / "crops.jsonl"
    crop_manifest.write_text("crop evidence\n")
    sample = CropSample(tmp_path / "crop.png", "AC", 0, "owned", "scene-1", None, "a" * 64, "1")
    metadata = {"preflight_status": "READY", "selected_checkpoint": str(model),
                "selected_checkpoint_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "split_manifest_sha256": {"test": hashlib.sha256(split.read_bytes()).hexdigest()}}
    with pytest.raises(ValueError, match="coverage"):
        validate_test_provenance(metadata, model, split, crop_manifest, [replace(sample, label="BACK")])
    split.write_text(split.read_text() + "\n")
    with pytest.raises(ValueError, match="split manifest"):
        validate_test_provenance(metadata, model, split, crop_manifest, [sample])
