"""Exact 53-way classifier evaluation; Pixel holdout belongs to B5."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .labels import LABELS


def _safe_div(a, b):
    return a / b if b else None


def evaluate_scores(truth, scores, *, split="test"):
    if split not in ("train", "val", "test"):
        raise ValueError("Pixel holdout evaluation is reserved for B5")
    truth, scores = list(truth), list(scores)
    if not truth or len(truth) != len(scores):
        raise ValueError("Evaluation needs matching nonempty truth and score rows")
    matrix = {label: {candidate: 0 for candidate in LABELS} for label in LABELS}
    confidence = []
    predictions = []
    for actual, row in zip(truth, scores):
        if actual not in LABELS or len(row) != 53 or any(type(v) not in (float, int) or not 0 <= v <= 1 for v in row):
            raise ValueError("Invalid label or 53-way probability row")
        index = max(range(53), key=lambda i: row[i])
        predicted = LABELS[index]
        predictions.append(predicted)
        confidence.append(float(row[index]))
        matrix[actual][predicted] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[other][label] for other in LABELS)
        per_class[label] = {"support": support, "recall": _safe_div(tp, support),
                            "precision": _safe_div(tp, predicted_count)}
    pairs = sorted(((actual, predicted, count) for actual in LABELS for predicted, count in matrix[actual].items()
                    if actual != predicted and count), key=lambda item: (-item[2], item[0], item[1]))
    thresholds = {}
    for threshold in (0.5, 0.7, 0.8, 0.9, 0.95):
        covered = [i for i, value in enumerate(confidence) if value >= threshold]
        thresholds[str(threshold)] = {"coverage": len(covered) / len(truth),
                                       "accuracy_at_coverage": _safe_div(sum(truth[i] == predictions[i] for i in covered), len(covered))}
    twos = ("2C", "2D", "2H", "2S")
    twos_matrix = {actual: {predicted: matrix[actual][predicted] for predicted in twos} for actual in twos}
    red, black = {"2D", "2H"}, {"2C", "2S"}
    red_black = sum(matrix[actual][predicted] for actual in red for predicted in black) + sum(
        matrix[actual][predicted] for actual in black for predicted in red)
    return {"schema_version": 1, "evaluated_split": split, "sample_count": len(truth),
            "top1_accuracy": sum(a == p for a, p in zip(truth, predictions)) / len(truth),
            "per_class": per_class, "confusion_matrix": matrix,
            "back": per_class["BACK"],
            "worst_classes": sorted(({"label": label, **values} for label, values in per_class.items() if values["support"]),
                                    key=lambda row: (row["recall"], row["label"]))[:10],
            "common_confusion_pairs": [{"actual": a, "predicted": p, "count": n} for a, p, n in pairs[:20]],
            "confidence": {"min": min(confidence), "mean": sum(confidence)/len(confidence), "max": max(confidence)},
            "threshold_coverage": thresholds,
            "twos": {"confusion_matrix": twos_matrix, "red_vs_black_confusions": red_black}}


def render_markdown(report):
    lines = ["# CardViper classifier evaluation", "", f"Split: **{report['evaluated_split']}**",
             f"Samples: {report['sample_count']}", f"Exact top-1: {report['top1_accuracy']:.4f}",
             f"BACK precision: {report['back']['precision']}", f"BACK recall: {report['back']['recall']}",
             f"Red/black two confusions: {report['twos']['red_vs_black_confusions']}", "",
             "## Worst classes", ""]
    lines.extend(f"- {row['label']}: recall {row['recall']:.4f}, support {row['support']}" for row in report["worst_classes"])
    lines.extend(["", "## Common confusions", ""])
    lines.extend(f"- {row['actual']} → {row['predicted']}: {row['count']}" for row in report["common_confusion_pairs"])
    lines.extend(["", "## Confidence / coverage", "", "| Threshold | Coverage | Accuracy |", "|---:|---:|---:|"])
    for threshold, value in report["threshold_coverage"].items():
        lines.append(f"| {threshold} | {value['coverage']:.4f} | {value['accuracy_at_coverage']} |")
    return "\n".join(lines) + "\n"


def evaluate_model(model_path, crop_directory, output_prefix, training_metadata):
    import numpy as np
    import tensorflow as tf
    from .classifier_dataset import load_crop_split, rgb_array
    metadata_path = Path(training_metadata)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("preflight_status") != "READY" or Path(metadata.get("selected_checkpoint", "")).resolve() != Path(model_path).resolve():
        raise ValueError("TEST evaluation requires a selected checkpoint from READY training")
    samples = load_crop_split(crop_directory, "test")
    model = tf.keras.models.load_model(model_path)
    size = int(model.input_shape[1])
    rows = np.asarray([np.asarray(rgb_array(sample, size), dtype=np.float32) for sample in samples])
    scores = model.predict(rows, verbose=0)
    report = evaluate_scores([sample.label for sample in samples], scores.tolist(), split="test")
    output_prefix = Path(output_prefix)
    report_path = output_prefix.with_suffix(".json")
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    output_prefix.with_suffix(".md").write_text(render_markdown(report))
    metadata["evaluation_report"] = str(report_path.resolve())
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--test-crops", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--training-metadata", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate_model(args.model, args.test_crops, args.output_prefix, args.training_metadata)
    print(f"Test top-1: {report['top1_accuracy']:.4f}")


if __name__ == "__main__":
    main()
