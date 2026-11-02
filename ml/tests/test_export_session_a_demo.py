"""Packaging boundaries for the isolated Session A Android assets."""

import hashlib
import json

import pytest

from cardviper_ml.labels import LABELS


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _classifier_report(path):
    return {
        "model_sha256": _sha256(path),
        "model_bytes": path.stat().st_size,
        "input_tensors": [{"name": "rgb_0_255", "shape": [1, 224, 224, 3], "dtype": "float32", "quantization": {"scales": []}}],
        "output_tensors": [{"name": "session_a_face_identity", "shape": [1, 52], "dtype": "float32", "quantization": {"scales": []}}],
    }


def _detector_report(path, *, valid=True):
    outputs = [
        {"name": "boxes", "shape": [1, 10, 4], "dtype": "float32"},
        {"name": "classes", "shape": [1, 10], "dtype": "float32"},
        {"name": "scores", "shape": [1, 10], "dtype": "float32"},
        {"name": "count", "shape": [1], "dtype": "float32"},
    ]
    return {
        "artifact_sha256": _sha256(path),
        "file_size_bytes": path.stat().st_size,
        "input_tensors": [{"name": "image_tensor", "shape": [1, 320, 320, 3], "dtype": "float32"}],
        "output_tensors": outputs if valid else outputs[:2],
        "contains_detection_postprocess_custom_op": valid,
    }


def _training_dirs(tmp_path):
    classifier = tmp_path / "classifier"
    detector = tmp_path / "detector"
    classifier.mkdir()
    detector.mkdir()
    classifier_model = classifier / "session_a_face_classifier.tflite"
    detector_model = detector / "session_a_card_detector.tflite"
    classifier_model.write_bytes(b"classifier-model")
    detector_model.write_bytes(b"detector-model")
    classifier.joinpath("training_metadata.json").write_text(json.dumps({
        "usage": "SESSION_A_DEMO_ONLY", "independence": "NOT_INDEPENDENT",
        "independent_evaluation": False, "face_up_only": True, "back_supported": False,
        "supported_labels": list(LABELS[:-1]), "output_width": 52,
        "preprocessing": "MobileNetV3Small include_preprocessing=True; RGB float32 0..255",
        "archive_sha256": "a" * 64,
    }))
    detector.joinpath("training_metadata.json").write_text(json.dumps({
        "usage": "SESSION_A_DEMO_ONLY", "independence": "NOT_INDEPENDENT",
        "independent_evaluation": False, "face_up_only": True, "back_supported": False,
        "semantic_classes": ["CARD"], "archive_sha256": "a" * 64,
    }))
    return classifier, detector, classifier_model, detector_model


def test_package_writes_hashed_52_face_assets(tmp_path):
    """A package that changes labels, hashes, or demo limits must fail this test."""
    classifier, detector, classifier_model, detector_model = _training_dirs(tmp_path)
    from cardviper_ml.export_session_a_demo import package_session_a_demo

    result = package_session_a_demo(
        classifier, detector, tmp_path / "assets",
        classifier_inspector=lambda _: _classifier_report(classifier_model),
        detector_inspector=lambda _: _detector_report(detector_model),
    )

    assets = tmp_path / "assets"
    assert result["usage"] == "SESSION_A_DEMO_ONLY"
    assert sorted(path.name for path in assets.iterdir()) == [
        "classifier.json", "classifier.tflite", "detector.json", "detector.tflite", "labels.txt", "manifest.json",
    ]
    assert assets.joinpath("labels.txt").read_text().splitlines() == list(LABELS[:-1])
    classifier_contract = json.loads(assets.joinpath("classifier.json").read_text())
    assert classifier_contract["model_kind"] == "SESSION_A_FACE_CLASSIFIER"
    assert classifier_contract["labels"] == list(LABELS[:-1])
    assert classifier_contract["outputs"][0]["shape"] == [1, 52]
    detector_contract = json.loads(assets.joinpath("detector.json").read_text())
    assert [tensor["name"] for tensor in detector_contract["outputs"]] == ["boxes", "classes", "scores", "count"]
    manifest = json.loads(assets.joinpath("manifest.json").read_text())
    assert manifest["back_supported"] is False
    assert manifest["assets"]["classifier"]["sha256"] == _sha256(assets / "classifier.tflite")


def test_package_rejects_detector_without_card_outputs(tmp_path):
    """A model without usable CARD boxes/classes/scores must never be packaged."""
    classifier, detector, classifier_model, detector_model = _training_dirs(tmp_path)
    from cardviper_ml.export_session_a_demo import package_session_a_demo

    with pytest.raises(ValueError, match="detector"):
        package_session_a_demo(
            classifier, detector, tmp_path / "assets",
            classifier_inspector=lambda _: _classifier_report(classifier_model),
            detector_inspector=lambda _: _detector_report(detector_model, valid=False),
        )
    assert not (tmp_path / "assets").exists()
