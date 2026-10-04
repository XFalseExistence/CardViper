"""Freeze classifier tensor and label mapping from a real inspected artifact."""

import argparse
import hashlib
import json
from pathlib import Path

from .labels import LABELS
from .inspect_litert import inspect_tflite


def build_contract(model_path, training_metadata, labels_path, evaluation_report, output_path=None):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    labels_path = Path(labels_path)
    raw_labels = labels_path.read_bytes()
    labels = raw_labels.decode("utf-8").splitlines()
    if labels != list(LABELS):
        raise ValueError("labels.txt differs from exact canonical index order")
    metadata = dict(training_metadata)
    if not metadata.get("preprocessing") or not metadata.get("architecture") or metadata.get("input_layout") != "NHWC":
        raise ValueError("Actual training preprocessing and architecture required")
    inspection = inspect_tflite(model_path)
    input_tensor, output_tensor = inspection["input_tensors"][0], inspection["output_tensors"][0]
    contract = {"schema_version": 1, "model_sha256": inspection["model_sha256"],
                "model_bytes": inspection["model_bytes"], "input_tensor": input_tensor,
                "output_tensor": output_tensor, "input_layout": metadata["input_layout"],
                "preprocessing": metadata["preprocessing"],
                "index_to_label": {str(i): label for i, label in enumerate(labels)},
                "back_index": 52, "labels_sha256": hashlib.sha256(raw_labels).hexdigest(),
                "evaluation_report": str(evaluation_report) if evaluation_report else None}
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
