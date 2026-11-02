"""Prepare deliberately non-promotable, face-up-only Session A demo data."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .audit_dataset_artifact import SOURCE_ID, audit_artifact
from .build_classifier_crops import build_crops
from .labels import LABELS
from .manifest import write_manifest
from .prepare_face_source import prepare_face_source
from .splits import record_key


DEMO_USAGE = "SESSION_A_DEMO_ONLY"
INDEPENDENCE = "NOT_INDEPENDENT"


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


def _require_current_audit(archive: Path, audit_path: Path) -> dict:
    """Ensure the supplied audit describes these exact archive bytes and labels."""
    audit = _read_json(audit_path, "artifact audit")
    expected_labels = {str(index): label for index, label in enumerate(LABELS[:-1])}
    if (audit.get("artifact_status") != "SOURCE_ARTIFACT_VALID" or
            audit.get("source_id") != SOURCE_ID or
            audit.get("archive_sha256") != _sha256(archive) or
            audit.get("class_index_to_label") != expected_labels):
        raise ValueError("Artifact audit is not approved for these Session A archive bytes")
    current = audit_artifact(archive)
    fields = (
        "artifact_status", "source_id", "archive_sha256", "data_yaml_sha256",
        "class_index_to_label", "source_class_index_to_name", "image_count",
        "annotation_count", "box_count",
    )
    if any(audit.get(field) != current.get(field) for field in fields):
        raise ValueError("Artifact audit does not match the current Session A archive")
    return audit


def _split_session_a_records(records):
    """Reserve a deterministic tail slice only for smoke checks, never metrics."""
    ordered = sorted(records, key=record_key)
    if len(ordered) < 2:
        raise ValueError("Session A demo needs at least two annotated images for train and smoke checks")
    smoke_count = min(max(1, len(ordered) // 10), len(ordered) - 1)
    train = ordered[:-smoke_count]
    smoke = [replace(row, held_out=True) for row in ordered[-smoke_count:]]
    return train, smoke


def _metadata(*, archive: Path, audit_path: Path, review_path: Path, source_root: Path,
              records, train, smoke, seed: int) -> dict:
    return {
        "schema_version": 1,
        "usage": DEMO_USAGE,
        "independence": INDEPENDENCE,
        "independent_evaluation": False,
        "face_up_only": True,
        "back_supported": False,
        "classifier_labels": list(LABELS[:-1]),
        "archive_sha256": _sha256(archive),
        "artifact_audit_sha256": _sha256(audit_path),
        "reviewed_groups_sha256": _sha256(review_path),
        "source_root": str(source_root),
        "seed": seed,
        "smoke_split": "deterministic source-path tail; not independent evaluation",
        "record_counts": {"all": len(records), "train": len(train), "smoke": len(smoke)},
    }


def prepare_session_a_demo(archive: Path, audit: Path, reviewed_groups: Path, output: Path, *, seed: int) -> dict:
    """Create a fresh, local-only Session A demo workspace from approved inputs."""
    archive, audit_path, review_path, output = map(Path, (archive, audit, reviewed_groups, output))
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not archive.is_file():
        raise ValueError("Session A archive must be a file")
    if output.exists():
        raise FileExistsError(f"Session A demo output already exists: {output}")
    _require_current_audit(archive, audit_path)

    dry_report, _, _ = prepare_face_source(archive, output.parent / "unused", review_path, dry_run=True)
    if dry_report.get("status") != "READY":
        reasons = "; ".join(dry_report.get("reasons", ()))
        raise ValueError(f"Reviewed Session A groups must be approved and current: {reasons}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        report, records, source_root = prepare_face_source(archive, staging / "source", review_path)
        if report.get("status") != "READY" or records is None or source_root is None:
            raise ValueError("Reviewed Session A groups must be approved and current")
        if any(obj.label not in LABELS[:-1] for row in records for obj in row.objects):
            raise ValueError("Session A demo accepts exactly the 52 canonical face labels")
        train, smoke = _split_session_a_records(records)
        write_manifest(staging / "demo_manifest.jsonl", records)
        write_manifest(staging / "train.jsonl", train)
        write_manifest(staging / "smoke.jsonl", smoke)
        roots = {SOURCE_ID: source_root}
        build_crops(train, roots, staging / "classifier_crops" / "train")
        build_crops(smoke, roots, staging / "classifier_crops" / "smoke")
        metadata = _metadata(
            archive=archive,
            audit_path=audit_path,
            review_path=review_path,
            source_root=source_root,
            records=records,
            train=train,
            smoke=smoke,
            seed=seed,
        )
        (staging / "demo_metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "READY", "output": str(output), "usage": DEMO_USAGE,
            "independence": INDEPENDENCE, "record_counts": metadata["record_counts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--reviewed-groups", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_session_a_demo(
        args.archive, args.audit, args.reviewed_groups, args.output, seed=args.seed),
        sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
