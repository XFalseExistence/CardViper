"""Integration gates for one-command B2 data onboarding."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pytest
from PIL import Image
import yaml

from cardviper_ml import prepare_classifier_data as onboarding
from cardviper_ml.manifest import Annotation, ImageRecord
from cardviper_ml.labels import LABELS

prepare_classifier_data = onboarding.prepare_classifier_data


SOURCES = Path(__file__).resolve().parents[1] / "datasets" / "sources.json"


def test_dry_run_reports_both_missing_sources_without_writing(tmp_path):
    workspace = tmp_path / "b2"
    report = prepare_classifier_data(
        tmp_path / "missing-face.zip", tmp_path / "missing-back", workspace,
        SOURCES, dry_run=True,
    )
    assert report["status"] == "NOT READY"
    assert {reason["code"] for reason in report["reasons"]} >= {
        "FACE_SOURCE_MISSING", "BACK_SOURCE_MISSING"
    }
    assert report["planned_outputs"]["splits"] == str(workspace / "splits")
    assert not workspace.exists()


def test_cli_dry_run_exits_not_ready_and_leaves_workspace_absent(tmp_path):
    workspace = tmp_path / "b2"
    process = subprocess.run([
        sys.executable, "-m", "cardviper_ml.prepare_classifier_data",
        "--face-source", str(tmp_path / "face.zip"),
        "--back-source", str(tmp_path / "back"),
        "--workspace", str(workspace),
        "--sources", str(SOURCES),
        "--dry-run",
    ], capture_output=True, text=True, check=False)
    assert process.returncode == 2
    assert json.loads(process.stdout)["status"] == "NOT READY"
    assert not workspace.exists()


def test_dry_run_reports_invalid_source_registry_without_writing(tmp_path):
    source_registry = tmp_path / "sources.json"
    source_registry.write_text('{"schema_version": 0, "sources": []}')
    workspace = tmp_path / "b2"
    report = prepare_classifier_data(
        tmp_path / "face", tmp_path / "back", workspace, source_registry,
        dry_run=True,
    )
    assert "SOURCE_REGISTRY_INVALID" in {reason["code"] for reason in report["reasons"]}
    assert not workspace.exists()


def test_dry_run_audits_existing_but_invalid_inputs(tmp_path):
    face, back, workspace = (tmp_path / name for name in ("face", "back", "b2"))
    face.mkdir()
    back.mkdir()
    report = prepare_classifier_data(face, back, workspace, SOURCES, dry_run=True)
    assert report["status"] == "NOT READY"
    assert {reason["code"] for reason in report["reasons"]} >= {
        "FACE_AUDIT_INVALID", "BACK_AUDIT_NOT_READY"
    }
    assert not workspace.exists()


def test_grouped_split_search_covers_each_required_identity():
    rows = []
    for label, source_id in (("AC", "playing-cards-seed"), ("BACK", "cardviper-back")):
        for number in range(6):
            rows.append(ImageRecord(
                source_id=source_id, scene_id=f"{label}-{number}",
                session_id=f"session-{label}-{number}",
                image_path=f"images/{label}-{number}.png",
                width=2, height=2,
                image_sha256=hashlib.sha256(f"{label}-{number}".encode()).hexdigest(),
                annotation_path=f"labels/{label}-{number}.txt",
                objects=(Annotation("1", label, (0, 0, 2, 2)),),
            ))
    splits, used_seed = onboarding.select_covering_split(rows, 42, required_labels={"AC", "BACK"})
    assert used_seed >= 42
    for name in ("train", "val", "test"):
        assert {obj.label for row in splits[name] for obj in row.objects} == {"AC", "BACK"}


def test_source_promotion_requires_evidence_and_keeps_detector_permission_off():
    locked = json.loads(SOURCES.read_text())
    face = {
        "status": "READY", "source_id": "playing-cards-seed",
        "artifact_audit": {"artifact_status": "SOURCE_ARTIFACT_VALID",
                           "archive_sha256": "a" * 64, "image_count": 12,
                           "annotation_count": 12,
                           "class_index_to_label": {str(i): label for i, label in enumerate(LABELS[:-1])}},
        "data_yaml_sha256": "b" * 64,
        "reviewed_groups_sha256": "c" * 64,
    }
    back = {
        "status": "READY", "source_id": "cardviper-back",
        "rights": {"classifier": True, "detector": True},
        "capture_manifest_sha256": "d" * 64,
        "sample_count": 8, "independent_sessions": 8,
    }
    with pytest.raises(ValueError):
        onboarding.promote_source_metadata(locked, {**face, "status": "NOT READY"}, back,
                                           retrieval_date="2026-10-04")
    promoted = onboarding.promote_source_metadata(locked, face, back,
                                                  retrieval_date="2026-10-04")
    face_source, back_source = promoted["sources"]
    assert face_source["verified"] and face_source["permitted_use"] == {
        "classifier": True, "detector": False
    }
    assert face_source["export"]["archive_sha256"] == "a" * 64
    assert back_source["verified"] and back_source["permitted_use"] == {
        "classifier": True, "detector": False
    }
    assert back_source["evidence"]["capture_manifest_sha256"] == "d" * 64
    assert locked["sources"][0]["verified"] is False


def _write_face_fixture(root):
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({
        "nc": 52, "names": list(LABELS[:-1]),
        "roboflow": {"workspace": "joshuas-workspace", "project": "playing-cards-9gfac", "version": 2},
    }))
    paths = []
    for number in range(6):
        split = ("train", "valid", "test")[number % 3]
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        name = f"scene-{number}_jpg.rf.{number:04x}"
        image = images / f"{name}.png"
        Image.new("RGB", (130, 40), (number * 25, 80, 150)).save(image)
        lines = []
        for class_id in range(52):
            column, row = class_id % 13, class_id // 13
            lines.append(f"{class_id} {(column * 10 + 5) / 130:.8f} {(row * 10 + 5) / 40:.8f} "
                         f"{8 / 130:.8f} {8 / 40:.8f}\n")
        (labels / f"{name}.txt").write_text("".join(lines))
        paths.append(image.relative_to(root).as_posix())
    return paths


def _write_back_fixture(root):
    root.mkdir()
    (root / "images").mkdir()
    captures = []
    for number in range(6):
        image = root / "images" / f"back-{number}.png"
        Image.new("RGB", (12, 12), (20 + number * 30, 10, 40)).save(image)
        captures.append({
            "capture_date": "2026-10-04", "device": "synthetic test camera",
            "session_id": f"session-{number}", "scene_id": f"scene-{number}",
            "deck_design": f"design-{number % 2}",
            "annotation_provenance": "Synthetic integration fixture only",
            "image_path": image.relative_to(root).as_posix(),
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "capture_mode": "tight-back-crop", "full_image_back": True,
        })
    (root / "captures.json").write_text(json.dumps({
        "schema_version": 1, "source_id": "cardviper-back", "owner": "Synthetic test fixture",
        "rights": {"classifier": True, "detector": False,
                   "evidence": "Synthetic fixture for isolated tests"},
        "captures": captures,
        "review": {"reviewer": "Synthetic test fixture", "notes": "Independent generated scenes",
                   "approved": True, "session_independence_confirmed": True,
                   "near_duplicate_clues_reviewed": True},
    }, sort_keys=True))


def test_valid_artifact_still_requires_explicit_face_family_review(tmp_path):
    face, back, workspace = (tmp_path / name for name in ("face", "back", "b2"))
    _write_face_fixture(face)
    _write_back_fixture(back)
    report = prepare_classifier_data(face, back, workspace, SOURCES, dry_run=True)
    assert report["face_audit"]["artifact_status"] == "SOURCE_ARTIFACT_VALID"
    assert report["back_audit"]["status"] == "READY"
    assert "FACE_GROUP_REVIEW_REQUIRED" in {reason["code"] for reason in report["reasons"]}
    assert report["status"] == "NOT READY"
    assert not workspace.exists()


def test_reviewed_sources_flow_through_splits_crops_and_ready_preflight(tmp_path):
    face, back, workspace = (tmp_path / name for name in ("face", "back", "b2"))
    paths = _write_face_fixture(face)
    _write_back_fixture(back)
    proposal, _, _ = onboarding.prepare_face_source(face, tmp_path / "unused", dry_run=True)
    groups = tmp_path / "reviewed-groups.json"
    groups.write_text(json.dumps({
        "schema_version": 1, "source_id": "playing-cards-seed",
        "artifact_fingerprint": proposal["group_proposal"]["artifact_fingerprint"],
        "groups": {path: f"independent-scene-{index}" for index, path in enumerate(paths)},
        "review": {"reviewer": "Synthetic fixture", "review_date": "2026-10-04",
                   "method": "Known independent generated source rasters",
                   "approved": True, "similarity_candidates_reviewed": True},
    }, sort_keys=True))
    registry = tmp_path / "sources.json"
    registry.write_bytes(SOURCES.read_bytes())
    report = prepare_classifier_data(
        face, back, workspace, registry, face_groups=groups,
        retrieval_date="2026-10-04",
    )
    assert report["status"] == "READY", report["reasons"]
    assert report["preflight"]["status"] == "READY"
    assert report["preflight"]["total_card_annotations"] == 318
    assert (workspace / "splits" / "test.jsonl").is_file()
    assert (workspace / "crops" / "train" / "crops.jsonl").is_file()
    assert json.loads(registry.read_text())["sources"][0]["verified"] is False
    runtime_sources = json.loads((workspace / "sources.json").read_text())
    assert all(source["permitted_use"]["detector"] is False for source in runtime_sources["sources"])
    promoted_workspace = tmp_path / "promoted-b2"
    promoted_report = prepare_classifier_data(
        face, back, promoted_workspace, registry, face_groups=groups,
        retrieval_date="2026-10-04", promote_sources=True,
    )
    assert promoted_report["status"] == "READY", promoted_report["reasons"]
    tracked_sources = json.loads(registry.read_text())
    assert all(source["verified"] for source in tracked_sources["sources"])
    assert all(source["permitted_use"]["detector"] is False for source in tracked_sources["sources"])


def test_failed_class_coverage_never_publishes_permission_registry(tmp_path):
    face, back, workspace = (tmp_path / name for name in ("face", "back", "b2"))
    paths = _write_face_fixture(face)
    _write_back_fixture(back)
    for label in face.rglob("*.txt"):
        label.write_text("".join(line for line in label.read_text().splitlines(keepends=True)
                                 if not line.startswith("51 ")))
    proposal, _, _ = onboarding.prepare_face_source(face, tmp_path / "unused", dry_run=True)
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(json.dumps({
        "schema_version": 1, "source_id": "playing-cards-seed",
        "artifact_fingerprint": proposal["group_proposal"]["artifact_fingerprint"],
        "groups": {path: f"scene-{index}" for index, path in enumerate(paths)},
        "review": {"reviewer": "Synthetic fixture", "review_date": "2026-10-04",
                   "method": "Generated independent scenes", "approved": True,
                   "similarity_candidates_reviewed": True},
    }))
    report = prepare_classifier_data(face, back, workspace, SOURCES,
                                     face_groups=reviewed, retrieval_date="2026-10-04")
    assert report["status"] == "NOT READY"
    assert report["preflight"]["status"] == "NOT READY"
    assert not (workspace / "sources.json").exists()
    assert not (workspace / "candidate-sources.json").exists()
