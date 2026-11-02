"""Tests for the intentionally non-promotable Session A demo data set."""

import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest
import yaml

from cardviper_ml.audit_dataset_artifact import audit_artifact
from cardviper_ml.labels import LABELS
from cardviper_ml.prepare_face_source import prepare_face_source


def _write_archive(tmp_path: Path) -> Path:
    """Create a minimal, source-shaped 52-face export without private data."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.yaml").write_text(yaml.safe_dump({
        "nc": 52,
        "names": list(LABELS[:-1]),
        "roboflow": {
            "workspace": "joshuas-workspace",
            "project": "playing-cards-9gfac",
            "version": 2,
        },
    }), encoding="utf-8")
    for index, label in enumerate(("AC", "2D")):
        image_dir = source / "train" / "images"
        label_dir = source / "train" / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        name = f"droidcam_20230224_205615_{index:04d}.png"
        Image.new("RGB", (16, 12), (index * 100, 20, 40)).save(image_dir / name)
        label_dir.joinpath(name).with_suffix(".txt").write_text(
            f"{LABELS.index(label)} 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    archive = tmp_path / "session-a.zip"
    with zipfile.ZipFile(archive, "w") as packed:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                packed.write(path, (Path("export") / path.relative_to(source)).as_posix())
    return archive


def _approved_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive = _write_archive(tmp_path)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit_artifact(archive)), encoding="utf-8")
    dry_report, _, _ = prepare_face_source(archive, tmp_path / "unused", dry_run=True)
    proposal = dry_report["group_proposal"]
    review_path = tmp_path / "reviewed-groups.json"
    review_path.write_text(json.dumps({
        "schema_version": 1,
        "source_id": "playing-cards-seed",
        "artifact_fingerprint": proposal["artifact_fingerprint"],
        "groups": {path: "session-a" for path in proposal["groups"]},
        "review": {
            "approved": True,
            "similarity_candidates_reviewed": True,
            "reviewer": "synthetic fixture",
            "review_date": "2026-11-01",
            "method": "synthetic source review",
        },
    }, sort_keys=True), encoding="utf-8")
    return archive, audit_path, review_path


def test_prepare_rejects_unapproved_review_without_creating_output(tmp_path):
    """A change that accepts an unapproved review must fail this test."""
    archive, audit_path, review_path = _approved_inputs(tmp_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["review"]["approved"] = False
    review_path.write_text(json.dumps(review), encoding="utf-8")

    from cardviper_ml.session_a_demo_data import prepare_session_a_demo

    output = tmp_path / "demo"
    with pytest.raises(ValueError, match="approved"):
        prepare_session_a_demo(archive, audit_path, review_path, output, seed=7)
    assert not output.exists()


def test_prepare_rejects_audit_bound_to_different_archive_bytes(tmp_path):
    """A stale audit must not authorize a changed source archive."""
    archive, audit_path, review_path = _approved_inputs(tmp_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["archive_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    from cardviper_ml.session_a_demo_data import prepare_session_a_demo

    with pytest.raises(ValueError, match="audit"):
        prepare_session_a_demo(archive, audit_path, review_path, tmp_path / "demo", seed=7)


def test_prepare_writes_deterministic_face_only_manifests_crops_and_metadata(tmp_path):
    """Dropping the face-only/provenance boundary must fail this end-to-end test."""
    archive, audit_path, review_path = _approved_inputs(tmp_path)

    from cardviper_ml.session_a_demo_data import prepare_session_a_demo

    output = tmp_path / "demo"
    result = prepare_session_a_demo(archive, audit_path, review_path, output, seed=7)

    assert result["status"] == "READY"
    assert result["usage"] == "SESSION_A_DEMO_ONLY"
    assert result["independence"] == "NOT_INDEPENDENT"
    assert (output / "demo_manifest.jsonl").is_file()
    assert (output / "train.jsonl").is_file()
    assert (output / "smoke.jsonl").is_file()
    assert (output / "classifier_crops" / "train" / "crops.jsonl").is_file()
    assert (output / "classifier_crops" / "smoke" / "crops.jsonl").is_file()
    metadata = json.loads((output / "demo_metadata.json").read_text(encoding="utf-8"))
    assert metadata["usage"] == "SESSION_A_DEMO_ONLY"
    assert metadata["independence"] == "NOT_INDEPENDENT"
    assert metadata["independent_evaluation"] is False
    assert metadata["face_up_only"] is True
    assert metadata["back_supported"] is False
    assert metadata["classifier_labels"] == list(LABELS[:-1])
    assert "BACK" not in metadata["classifier_labels"]
    assert metadata["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert result["record_counts"] == {"all": 2, "train": 1, "smoke": 1}


def test_prepare_refuses_to_overwrite_an_existing_output(tmp_path):
    """A change that overwrites a prior demo preparation must fail this test."""
    archive, audit_path, review_path = _approved_inputs(tmp_path)

    from cardviper_ml.session_a_demo_data import prepare_session_a_demo

    output = tmp_path / "demo"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_session_a_demo(archive, audit_path, review_path, output, seed=7)
