"""Freeze classifier tensor and label mapping from a real inspected artifact."""

import argparse
import hashlib
import json
from pathlib import Path

from .labels import LABELS
from .inspect_litert import inspect_tflite
from .classifier_evidence import validate_export_evidence


def build_contract(model_path, training_metadata, labels_path, evaluation_report, output_path=None, *, smoke=False):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    labels_path = Path(labels_path)
    raw_labels = labels_path.read_bytes()
    labels = raw_labels.decode("utf-8").splitlines()
    if labels != list(LABELS):
        raise ValueError("labels.txt differs from exact canonical index order")
    metadata = dict(training_metadata)
    if (metadata.get("preprocessing") != "MobileNetV3Small include_preprocessing=True; RGB float32 0..255" or
            metadata.get("architecture") != "MobileNetV3Small" or metadata.get("input_layout") != "NHWC"):
        raise ValueError("Actual training preprocessing and architecture required")
    if not smoke:
        if str(evaluation_report) != str(metadata.get("evaluation_report")):
            raise ValueError("Evaluation report differs from training metadata")
        validate_export_evidence(metadata, metadata.get("selected_checkpoint", ""))
    elif evaluation_report != "SYNTHETIC-SMOKE-ONLY":
        raise ValueError("Synthetic contract must be explicitly labeled")
    inspection = inspect_tflite(model_path)
    input_tensor, output_tensor = inspection["input_tensors"][0], inspection["output_tensors"][0]
    size = metadata.get("input_size")
    if (input_tensor["shape"][1:] != [size, size, 3] or input_tensor["dtype"] != "float32" or
            output_tensor["dtype"] != "float32" or
            input_tensor["quantization"]["scales"] or output_tensor["quantization"]["scales"]):
        raise ValueError("Exported tensor shape/dtype/quantization differs from float training contract")
    contract = {"schema_version": 1, "model_sha256": inspection["model_sha256"],
                "model_bytes": inspection["model_bytes"], "input_tensor": input_tensor,
                "output_tensor": output_tensor, "input_layout": metadata["input_layout"],
                "preprocessing": metadata["preprocessing"],
                "index_to_label": {str(i): label for i, label in enumerate(labels)},
                "back_index": 52, "labels_sha256": hashlib.sha256(raw_labels).hexdigest(),
                "evaluation_report": str(evaluation_report) if evaluation_report else None,
                "synthetic_smoke": bool(smoke)}
    if output_path:
        Path(output_path).write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.training_metadata.read_text())
    build_contract(args.model, metadata, args.labels, args.evaluation_report, args.output)


if __name__ == "__main__":
    main()
