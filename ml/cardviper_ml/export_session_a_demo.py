"""Inspect and package verified, local-only Session A phone-demo LiteRT assets."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .labels import LABELS


DEMO_USAGE = "SESSION_A_DEMO_ONLY"
INDEPENDENCE = "NOT_INDEPENDENT"
DEMO_LABELS = LABELS[:-1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {description}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {description}")
    return value


def _require_demo_metadata(path: Path, *, classifier: bool) -> dict:
    value = _read_json(path / "training_metadata.json", "Session A training metadata")
    if (value.get("usage") != DEMO_USAGE or value.get("independence") != INDEPENDENCE or
            value.get("independent_evaluation") is not False or value.get("face_up_only") is not True or
            value.get("back_supported") is not False):
        raise ValueError("Training metadata is not an approved Session A-only demo artifact")
    if classifier:
        if value.get("supported_labels") != list(DEMO_LABELS) or value.get("output_width") != 52:
            raise ValueError("Classifier metadata is not an exact 52-face Session A contract")
    elif value.get("semantic_classes") != ["CARD"]:
        raise ValueError("Detector metadata does not declare exactly one CARD class")
    archive_sha = value.get("archive_sha256")
    if not isinstance(archive_sha, str) or len(archive_sha) != 64:
        raise ValueError("Training metadata is missing the source archive SHA-256")
    return value


def _convert_classifier_if_needed(directory: Path) -> Path:
    direct = directory / "session_a_face_classifier.tflite"
    if direct.is_file():
        return direct
    keras = directory / "session_a_face_demo.keras"
    if not keras.is_file():
        raise ValueError("Classifier artifact is missing")
    import tensorflow as tf

    model = tf.keras.models.load_model(keras)
    if model.output_shape != (None, 52):
        raise ValueError("Session A classifier Keras model is not 52-face")
    with tempfile.TemporaryDirectory() as workspace:
        model.export(workspace)
        converter = tf.lite.TFLiteConverter.from_saved_model(workspace)
        direct.write_bytes(converter.convert())
    return direct


def inspect_session_a_classifier(path: Path) -> dict:
    """Inspect an actual 52-face LiteRT flatbuffer without applying the 53-way rules."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError as error:
            raise RuntimeError("Install the train extra or ai-edge-litert to inspect a real classifier") from error
    import numpy as np

    content = Path(path).read_bytes()
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    def tensor(detail):
        quant = detail.get("quantization_parameters", {})
        return {
            "name": detail["name"],
            "shape": [int(value) for value in detail["shape"]],
            "dtype": np.dtype(detail["dtype"]).name,
            "quantization": {"scales": [float(value) for value in quant.get("scales", [])]},
        }

    return {
        "model_sha256": hashlib.sha256(content).hexdigest(),
        "model_bytes": len(content),
        "input_tensors": [tensor(detail) for detail in interpreter.get_input_details()],
        "output_tensors": [tensor(detail) for detail in interpreter.get_output_details()],
    }


def _classifier_contract(report: dict, model: Path, labels_bytes: bytes) -> dict:
    inputs = report.get("input_tensors")
    outputs = report.get("output_tensors")
    if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("classifier must expose exactly one input and one score output")
    input_tensor, output_tensor = inputs[0], outputs[0]
    if (report.get("model_sha256") != _sha256(model) or report.get("model_bytes") != model.stat().st_size or
            input_tensor.get("shape") != [1, 224, 224, 3] or input_tensor.get("dtype") != "float32" or
            output_tensor.get("shape") != [1, 52] or output_tensor.get("dtype") != "float32" or
            input_tensor.get("quantization", {}).get("scales") or output_tensor.get("quantization", {}).get("scales")):
        raise ValueError("classifier inspection does not match the 52-face float LiteRT contract")
    return {
        "schema_version": 1,
        "model_kind": "SESSION_A_FACE_CLASSIFIER",
        "artifact": {"sha256": report["model_sha256"]},
        "input": {"name": input_tensor["name"], "shape": input_tensor["shape"], "layout": "NHWC", "dtype": "FLOAT32"},
        "preprocessing": {"resize_policy": "STRETCH", "resize_filter": "BILINEAR", "channel_order": "RGB", "scale": 1.0, "offset": 0.0, "pad_value": 0},
        "outputs": [{"name": output_tensor["name"], "role": "CLASSIFIER_SCORES", "shape": output_tensor["shape"], "dtype": "FLOAT32"}],
        "labels": list(DEMO_LABELS),
        "labels_sha256": hashlib.sha256(labels_bytes).hexdigest(),
        "classifier": {"top_k": 3, "score_semantics": "PROBABILITIES"},
        "session_a_demo": {"usage": DEMO_USAGE, "independence": INDEPENDENCE, "face_up_only": True, "back_supported": False},
    }


def _detector_contract(report: dict, model: Path) -> dict:
    inputs = report.get("input_tensors")
    outputs = report.get("output_tensors")
    if (report.get("artifact_sha256") != _sha256(model) or report.get("file_size_bytes") != model.stat().st_size or
            report.get("contains_detection_postprocess_custom_op") is not True or
            not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(outputs, list) or len(outputs) != 4):
        raise ValueError("detector inspection does not expose the required CARD outputs")
    input_tensor = inputs[0]
    if input_tensor.get("shape") != [1, 320, 320, 3] or input_tensor.get("dtype") != "float32":
        raise ValueError("detector input is incompatible with the 320x320 CARD contract")
    boxes, classes, scores, count = outputs
    capacity = boxes.get("shape", [0, 0])[1] if isinstance(boxes.get("shape"), list) and len(boxes["shape"]) == 3 else 0
    if (boxes.get("shape") != [1, capacity, 4] or classes.get("shape") != [1, capacity] or
            scores.get("shape") != [1, capacity] or count.get("shape") != [1] or
            any(tensor.get("dtype") != "float32" for tensor in outputs)):
        raise ValueError("detector CARD output tensor surface is incompatible")
    return {
        "schema_version": 1,
        "model_kind": "DETECTOR",
        "artifact": {"sha256": report["artifact_sha256"]},
        "input": {"name": input_tensor["name"], "shape": input_tensor["shape"], "layout": "NHWC", "dtype": "FLOAT32"},
        "preprocessing": {"resize_policy": "STRETCH", "resize_filter": "BILINEAR", "channel_order": "RGB", "scale": 1.0, "offset": 0.0, "pad_value": 0},
        "outputs": [
            {"name": tensor["name"], "role": {
                boxes["name"]: "DETECTION_BOXES",
                scores["name"]: "DETECTION_SCORES",
                classes["name"]: "DETECTION_CLASSES",
                count["name"]: "DETECTION_COUNT",
            }.get(tensor["name"]), "shape": tensor["shape"], "dtype": "FLOAT32"}
            for tensor in outputs
        ],
        "labels": None,
        "labels_sha256": None,
        "classifier": None,
        "detector": {
            "semantic_classes": ["CARD"], "card_class_index": 0,
            "boxes": {"tensor": boxes["name"], "coordinate_order": "YXYX", "coordinate_units": "NORMALIZED"},
            "scores": {"tensor": scores["name"]}, "classes": {"tensor": classes["name"]},
            "count": {"tensor": count["name"]},
            "nms": {"ownership": "EMBEDDED", "score_threshold": 0.5, "iou_threshold": 0.5},
        },
    }


def package_session_a_demo(classifier_dir: Path, detector_dir: Path, destination: Path, *,
                           classifier_inspector=None, detector_inspector=None) -> dict:
    """Atomically write only verified Session A assets to a fresh local directory."""
    classifier_dir, detector_dir, destination = map(Path, (classifier_dir, detector_dir, destination))
    if destination.exists():
        raise FileExistsError(f"Session A asset destination already exists: {destination}")
    classifier_metadata = _require_demo_metadata(classifier_dir, classifier=True)
    detector_metadata = _require_demo_metadata(detector_dir, classifier=False)
    if classifier_metadata["archive_sha256"] != detector_metadata["archive_sha256"]:
        raise ValueError("Classifier and detector source archive bindings differ")
    classifier_model = _convert_classifier_if_needed(classifier_dir)
    detector_model = detector_dir / "session_a_card_detector.tflite"
    if not detector_model.is_file():
        raise ValueError("Detector artifact is missing")
    classifier_report = (classifier_inspector or inspect_session_a_classifier)(classifier_model)
    if detector_inspector is None:
        from .inspect_detector_litert import inspect_artifact
        detector_inspector = inspect_artifact
    detector_report = detector_inspector(detector_model)
    labels = ("\n".join(DEMO_LABELS) + "\n").encode("utf-8")
    classifier_contract = _classifier_contract(classifier_report, classifier_model, labels)
    detector_contract = _detector_contract(detector_report, detector_model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        shutil.copyfile(classifier_model, staging / "classifier.tflite")
        shutil.copyfile(detector_model, staging / "detector.tflite")
        (staging / "labels.txt").write_bytes(labels)
        (staging / "classifier.json").write_text(json.dumps(classifier_contract, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (staging / "detector.json").write_text(json.dumps(detector_contract, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "usage": DEMO_USAGE,
            "independence": INDEPENDENCE,
            "face_up_only": True,
            "back_supported": False,
            "source_archive_sha256": classifier_metadata["archive_sha256"],
            "assets": {
                "classifier": {"file": "classifier.tflite", "sha256": _sha256(staging / "classifier.tflite"), "contract": "classifier.json", "labels": "labels.txt"},
                "detector": {"file": "detector.tflite", "sha256": _sha256(staging / "detector.tflite"), "contract": "detector.json"},
            },
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "READY", "output": str(destination), "usage": DEMO_USAGE,
            "classifier_sha256": manifest["assets"]["classifier"]["sha256"],
            "detector_sha256": manifest["assets"]["detector"]["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-dir", required=True, type=Path)
    parser.add_argument("--detector-dir", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(package_session_a_demo(args.classifier_dir, args.detector_dir, args.destination), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
