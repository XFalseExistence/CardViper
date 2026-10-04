"""Evidence-gated B2 classifier data onboarding."""

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .build_classifier_crops import build_crops
from .classifier_preflight import audit_classifier_data, render_markdown
from .import_back_captures import audit_back_source, import_back_source
from .labels import LABELS
from .manifest import read_sources, write_manifest
from .prepare_face_source import prepare_face_source
from .splits import SPLIT_NAMES, split_records, unique_records, write_splits


def select_covering_split(rows, seed=42, *, required_labels=LABELS, attempts=256):
    """Choose a deterministic grouped split with the best class coverage."""
    required = set(required_labels)
    if not required or type(attempts) is not int or attempts <= 0:
        raise ValueError("Required labels and positive attempts are needed")
    best = None
    best_score = -1
    for offset in range(attempts):
        candidate = split_records(rows, seed=seed + offset)
        score = sum(len({obj.label for row in candidate[name] for obj in row.objects} & required)
                    for name in ("train", "val", "test"))
        if score > best_score:
            best, best_score = (candidate, seed + offset), score
        if score == 3 * len(required):
            break
    return best


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def promote_source_metadata(locked, face, back, *, retrieval_date):
    """Build a portable classifier-only registry from reviewed local evidence."""
    if (not isinstance(retrieval_date, str) or
            date.fromisoformat(retrieval_date).isoformat() != retrieval_date):
        raise ValueError("An exact YYYY-MM-DD Joshua retrieval date is required")
    artifact = face.get("artifact_audit", {})
    mapped = artifact.get("class_index_to_label", {})
    if (face.get("status") != "READY" or face.get("source_id") != "playing-cards-seed" or
            artifact.get("artifact_status") != "SOURCE_ARTIFACT_VALID" or
            set(mapped) != {str(i) for i in range(52)} or
            set(mapped.values()) != set(LABELS[:-1]) or
            not _sha256(face.get("data_yaml_sha256")) or
            not _sha256(face.get("reviewed_groups_sha256"))):
        raise ValueError("Joshua artifact and grouping evidence are not ready for promotion")
    archive_sha = artifact.get("archive_sha256")
    if archive_sha is not None and not _sha256(archive_sha):
        raise ValueError("Invalid Joshua archive SHA-256")
    if (back.get("status") != "READY" or back.get("source_id") != "cardviper-back" or
            back.get("rights", {}).get("classifier") is not True or
            not _sha256(back.get("capture_manifest_sha256")) or
            back.get("sample_count", 0) <= 0 or back.get("independent_sessions", 0) < 3):
        raise ValueError("CardViper BACK rights and capture evidence are not ready for promotion")
    result = json.loads(json.dumps(locked))
    if result.get("schema_version") != 1:
        raise ValueError("Invalid source registry schema")
    face_source = next((item for item in result["sources"] if item["source_id"] == "playing-cards-seed"), None)
    if not face_source or face_source.get("version") != "2" or face_source.get("license") != "CC-BY-4.0":
        raise ValueError("Locked Joshua v2 source metadata is missing")
    face_source["artifact_verified"] = True
    face_source["grouping_verified"] = True
    face_source["verified"] = True
    face_source["permitted_use"] = {"classifier": True, "detector": False}
    face_source["export"].update({"format": "YOLOv8", "archive_sha256": archive_sha,
                                  "class_index_order_verified": True})
    face_source["evidence"] = {
        "artifact_export_type": "YOLOv8 ZIP" if archive_sha else "extracted YOLOv8",
        "data_yaml_sha256": face["data_yaml_sha256"],
        "class_index_order_sha256": _class_order_sha256(mapped),
        "reviewed_groups_sha256": face["reviewed_groups_sha256"],
        "retrieval_date": retrieval_date,
        "image_count": artifact["image_count"],
        "annotation_count": artifact["annotation_count"],
        "audit_report": "joshua-audit.json",
    }
    back_source = {
        "source_id": "cardviper-back", "name": "CardViper-owned BACK captures",
        "url": "local:cardviper-back", "license": "CardViper-owned",
        "attribution": "CardViper-owned capture manifest",
        "local_root": "CARDVIPER_BACK_ROOT", "verified": True,
        "permitted_use": {"classifier": True, "detector": False},
        "notes": "Classifier BACK samples; detector permission requires a separate review.",
        "evidence": {"capture_manifest_sha256": back["capture_manifest_sha256"],
                     "sample_count": back["sample_count"],
                     "independent_sessions": back["independent_sessions"],
                     "audit_report": "back-audit.json"},
    }
    result["sources"] = [item for item in result["sources"] if item["source_id"] != "cardviper-back"] + [back_source]
    return result


def _class_order_sha256(index_to_label):
    ordered = [index_to_label[str(i)] for i in range(52)]
    return hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def prepare_classifier_data(face_source, back_source, workspace, sources_path, *,
                            face_groups=None, class_map=None, capture_mode=None,
                            retrieval_date=None, dry_run=False,
                            promote_sources=False, seed=42):
    face_source = Path(face_source)
    back_source = Path(back_source)
    workspace = Path(workspace)
    sources_path = Path(sources_path)
    reasons = []
    report = {
        "schema_version": 1, "status": "NOT READY", "reasons": reasons,
        "face_audit": None, "face_review": None, "back_audit": None,
        "preflight": None, "selected_split_seed": None,
        "planned_outputs": {
            "manifest": str(workspace / "all.jsonl"),
            "splits": str(workspace / "splits"),
            "crops": str(workspace / "crops"),
            "roots": str(workspace / "roots.json"),
            "sources": str(workspace / "sources.json"),
            "preflight": str(workspace / "classifier-preflight.json"),
        },
    }
    if not face_source.exists():
        reasons.append({"code": "FACE_SOURCE_MISSING", "message": f"Joshua v2 export absent: {face_source}"})
    if not back_source.exists():
        reasons.append({"code": "BACK_SOURCE_MISSING", "message": f"CardViper BACK captures absent: {back_source}"})
    try:
        sources = read_sources(sources_path)
        face_metadata = sources["playing-cards-seed"]
        if (face_metadata.get("version") != "2" or
                face_metadata.get("project", {}).get("workspace_slug") != "joshuas-workspace" or
                face_metadata.get("project", {}).get("project_slug") != "playing-cards-9gfac" or
                face_metadata.get("license") != "CC-BY-4.0" or
                face_metadata.get("source_page_verified") is not True):
            raise ValueError("Locked Joshua v2 source identity or license differs")
    except (OSError, ValueError, KeyError, TypeError) as error:
        reasons.append({"code": "SOURCE_REGISTRY_INVALID", "message": str(error)})
    if face_source.exists():
        face_review, _, _ = prepare_face_source(
            face_source, workspace / "sources" / "playing-cards-seed",
            face_groups, dry_run=True, class_map_path=class_map,
        )
        report["face_review"] = face_review
        report["face_audit"] = face_review["artifact_audit"]
        if not report["face_audit"] or report["face_audit"]["artifact_status"] != "SOURCE_ARTIFACT_VALID":
            reasons.append({"code": "FACE_AUDIT_INVALID", "message": "; ".join(face_review["reasons"])})
        elif face_groups is None or not Path(face_groups).is_file():
            reasons.append({"code": "FACE_GROUP_REVIEW_REQUIRED",
                            "message": "Supply a reviewed Joshua family-group map with --face-groups"})
        elif face_review["status"] != "READY":
            reasons.append({"code": "FACE_GROUP_REVIEW_INVALID", "message": "; ".join(face_review["reasons"])})
    if back_source.exists():
        try:
            report["back_audit"] = audit_back_source(back_source, capture_mode=capture_mode)
            if report["back_audit"]["status"] != "READY":
                reasons.append({"code": "BACK_AUDIT_NOT_READY",
                                "message": "; ".join(report["back_audit"]["reasons"])})
        except (OSError, ValueError, KeyError, TypeError) as error:
            reasons.append({"code": "BACK_AUDIT_NOT_READY", "message": str(error)})
    if reasons:
        return report
    if dry_run:
        reasons.append({"code": "DRY_RUN_NO_PREFLIGHT",
                        "message": "Dry-run audits inputs but never creates splits or grants training permission"})
        return report
    if workspace.exists():
        reasons.append({"code": "WORKSPACE_EXISTS", "message": f"Use a new workspace path: {workspace}"})
        return report
    try:
        locked = json.loads(sources_path.read_text(encoding="utf-8"))
        locked_sha = hashlib.sha256(sources_path.read_bytes()).hexdigest()
        workspace.mkdir(parents=True, exist_ok=False)
        prepared_face, face_rows, face_root = prepare_face_source(
            face_source, workspace / "sources" / "playing-cards-seed",
            face_groups, class_map_path=class_map,
        )
        if prepared_face["status"] != "READY" or not face_rows:
            raise ValueError("Joshua normalization failed: " + "; ".join(prepared_face["reasons"]))
        if (prepared_face["data_yaml_sha256"] != face_review["data_yaml_sha256"] or
                prepared_face["reviewed_groups_sha256"] != face_review["reviewed_groups_sha256"] or
                prepared_face["group_proposal"]["artifact_fingerprint"] != face_review["group_proposal"]["artifact_fingerprint"] or
                prepared_face["artifact_audit"].get("archive_sha256") != face_review["artifact_audit"].get("archive_sha256")):
            raise ValueError("Joshua artifact or reviewed grouping changed during preparation")
        current_back = audit_back_source(back_source, capture_mode=capture_mode)
        if (current_back["status"] != "READY" or
                current_back["capture_manifest_sha256"] != report["back_audit"]["capture_manifest_sha256"]):
            raise ValueError("BACK capture evidence changed during preparation")
        back_rows = import_back_source(back_source, capture_mode=capture_mode)
        combined = face_rows + back_rows
        unique_records(combined)
        promoted = promote_source_metadata(locked, prepared_face, current_back,
                                           retrieval_date=retrieval_date)
        _write_json(workspace / "joshua-audit.json", prepared_face)
        _write_json(workspace / "back-audit.json", current_back)
        roots = {"playing-cards-seed": str(face_root),
                 "cardviper-back": str(back_source.resolve())}
        _write_json(workspace / "roots.json", roots)
        write_manifest(workspace / "all.jsonl", combined)
        splits, chosen_seed = select_covering_split(combined, seed=seed)
        report["selected_split_seed"] = chosen_seed
        write_splits(workspace / "splits", splits)
        crop_root = workspace / "crops"
        crop_root.mkdir()
        for name in SPLIT_NAMES:
            build_crops(splits[name], roots, crop_root / name, padding=0.1)
        with tempfile.TemporaryDirectory(prefix="cardviper-preflight-") as candidate_directory:
            candidate_sources = Path(candidate_directory) / "sources.json"
            _write_json(candidate_sources, promoted)
            preflight = audit_classifier_data(candidate_sources, workspace / "splits")
        report["preflight"] = preflight
        _write_json(workspace / "classifier-preflight.json", preflight)
        (workspace / "classifier-preflight.md").write_text(render_markdown(preflight), encoding="utf-8")
        report["face_review"] = prepared_face
        report["face_audit"] = prepared_face["artifact_audit"]
        report["back_audit"] = current_back
        if preflight["ready"]:
            report["status"] = "READY"
            _write_json(workspace / "sources.json", promoted)
            if promote_sources:
                if hashlib.sha256(sources_path.read_bytes()).hexdigest() != locked_sha:
                    raise ValueError("Tracked source registry changed during preparation")
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=sources_path.parent,
                                                 prefix=".cardviper-sources-", delete=False) as temporary:
                    temporary.write(json.dumps(promoted, sort_keys=True, indent=2) + "\n")
                    temporary_path = Path(temporary.name)
                os.replace(temporary_path, sources_path)
        else:
            reasons.extend(preflight["reasons"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["status"] = "NOT READY"
        reasons.append({"code": "PREPARATION_FAILED", "message": str(error)})
    if workspace.is_dir():
        _write_json(workspace / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-source", required=True, type=Path)
    parser.add_argument("--back-source", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--face-groups", type=Path)
    parser.add_argument("--class-map", type=Path, help="Explicit Joshua export class aliases JSON")
    parser.add_argument("--capture-mode", choices=("tight-back-crop", "annotated-scene"))
    parser.add_argument("--retrieval-date", help="YYYY-MM-DD date the Joshua export was obtained")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--promote-sources", action="store_true")
    args = parser.parse_args()
    report = prepare_classifier_data(args.face_source, args.back_source, args.workspace,
                                     args.sources, face_groups=args.face_groups,
                                     class_map=args.class_map, capture_mode=args.capture_mode,
                                     retrieval_date=args.retrieval_date, dry_run=args.dry_run,
                                     promote_sources=args.promote_sources, seed=args.seed)
    print(json.dumps(report, sort_keys=True, indent=2))
    raise SystemExit(0 if report["status"] == "READY" else 2)


if __name__ == "__main__":
    main()
