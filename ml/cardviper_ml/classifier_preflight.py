"""Fail-closed readiness audit for CardViper's 53-way classifier data."""

import argparse
import json
import statistics
from pathlib import Path

from .labels import LABELS
from .manifest import read_sources, require_training_permission
from .splits import SPLIT_NAMES, connected_groups, read_splits


def _empty_report(reason):
    return {
        "schema_version": 1,
        "status": "NOT READY",
        "ready": False,
        "reasons": [reason],
        "total_images": 0,
        "total_card_annotations": 0,
        "independent_groups": 0,
        "split_image_counts": {name: 0 for name in SPLIT_NAMES},
        "split_group_counts": {name: 0 for name in SPLIT_NAMES},
        "per_class_annotation_counts": {
            label: {**{name: 0 for name in SPLIT_NAMES}, "total": 0}
            for label in LABELS
        },
        "back_count": 0,
        "examples_per_class": {"min": 0, "median": 0, "max": 0},
        "permitted_source_ids": [],
        "held_out_count": 0,
    }


def audit_classifier_data(sources_path, splits_path):
    """Return a deterministic JSON-compatible readiness report; never throw on bad input."""
    try:
        sources = read_sources(Path(sources_path))
    except Exception as error:
        return _empty_report({"code": "SOURCE_MANIFEST_INVALID", "message": str(error)})
    try:
        splits = read_splits(Path(splits_path))
    except Exception as error:
        return _empty_report({"code": "SPLIT_INVALID", "message": str(error)})

    rows = [row for name in SPLIT_NAMES for row in splits[name]]
    source_ids = sorted({row.source_id for row in rows})
    reasons = []
    permitted = []
    for source_id in source_ids:
        try:
            require_training_permission(sources, source_id, "classifier")
            permitted.append(source_id)
        except ValueError as error:
            reasons.append({
                "code": "SOURCE_NOT_PERMITTED",
                "source_id": source_id,
                "message": str(error),
            })

    counts = {
        label: {**{name: 0 for name in SPLIT_NAMES}, "total": 0}
        for label in LABELS
    }
    for split_name in SPLIT_NAMES:
        for row in splits[split_name]:
            for annotation in row.objects:
                # ImageRecord construction already rejects unknown labels. Keep this
                # check so the auditor's invariant is explicit if representations evolve.
                if annotation.label not in counts:
                    reasons.append({
                        "code": "UNKNOWN_LABEL",
                        "label": annotation.label,
                        "message": f"Unknown classifier label: {annotation.label}",
                    })
                    continue
                counts[annotation.label][split_name] += 1
                counts[annotation.label]["total"] += 1

    missing = [label for label in LABELS if counts[label]["total"] == 0]
    if missing:
        reasons.append({
            "code": "REQUIRED_LABEL_MISSING",
            "labels": missing,
            "message": "Required labels absent from all data: " + ", ".join(missing),
        })
    for split_name, code in (
        ("train", "ZERO_TRAIN_CLASS"),
        ("val", "VALIDATION_CLASS_MISSING"),
        ("test", "TEST_CLASS_MISSING"),
    ):
        absent = [label for label in LABELS if counts[label][split_name] == 0]
        if absent:
            reasons.append({
                "code": code,
                "labels": absent,
                "message": f"{split_name} has zero examples for: " + ", ".join(absent),
            })

    totals = [counts[label]["total"] for label in LABELS]
    split_images = {name: len(splits[name]) for name in SPLIT_NAMES}
    split_groups = {name: len(connected_groups(splits[name])) for name in SPLIT_NAMES}
    report = {
        "schema_version": 1,
        "status": "READY" if not reasons else "NOT READY",
        "ready": not reasons,
        "reasons": reasons,
        "total_images": len(rows),
        "total_card_annotations": sum(totals),
        "independent_groups": len(connected_groups(rows)),
        "split_image_counts": split_images,
        "split_group_counts": split_groups,
        "per_class_annotation_counts": counts,
        "back_count": counts["BACK"]["total"],
        "examples_per_class": {
            "min": min(totals),
            "median": statistics.median(totals),
            "max": max(totals),
        },
        "permitted_source_ids": permitted,
        "held_out_count": split_images["holdout"],
    }
    return report


def render_markdown(report):
    lines = [
        "# CardViper classifier data preflight",
        "",
        f"Status: **{report['status']}**",
        "",
        f"- Total images: {report['total_images']}",
        f"- Total card annotations: {report['total_card_annotations']}",
        f"- Independent groups: {report['independent_groups']}",
        f"- Held-out images: {report['held_out_count']}",
        f"- BACK annotations: {report['back_count']}",
        f"- Examples per class (min / median / max): "
        f"{report['examples_per_class']['min']} / "
        f"{report['examples_per_class']['median']} / "
        f"{report['examples_per_class']['max']}",
        f"- Permitted sources: {', '.join(report['permitted_source_ids']) or 'none'}",
        "",
        "## Split coverage",
        "",
        "| Split | Images | Groups |",
        "|---|---:|---:|",
    ]
    for name in SPLIT_NAMES:
        lines.append(
            f"| {name} | {report['split_image_counts'][name]} | "
            f"{report['split_group_counts'][name]} |"
        )
    lines.extend(["", "## Reasons", ""])
    if report["reasons"]:
        lines.extend(
            f"- `{reason['code']}`: {reason['message']}"
            for reason in report["reasons"]
        )
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Per-class annotations",
        "",
        "| Label | Train | Validation | Test | Holdout | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label in LABELS:
        value = report["per_class_annotation_counts"][label]
        lines.append(
            f"| {label} | {value['train']} | {value['val']} | {value['test']} | "
            f"{value['holdout']} | {value['total']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--json", type=Path, help="Optional machine-readable report path")
    parser.add_argument("--markdown", type=Path, help="Optional human-readable report path")
    args = parser.parse_args()
    report = audit_classifier_data(args.sources, args.splits)
    machine = json.dumps(report, indent=2, sort_keys=True) + "\n"
    human = render_markdown(report)
    if args.json:
        args.json.write_text(machine, encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(human, encoding="utf-8")
    print(human, end="")
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
