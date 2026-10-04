"""Joshua v2 import must keep upstream variants grouped before resplitting."""

import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image
import yaml

from cardviper_ml.labels import LABELS
from cardviper_ml import prepare_face_source as face


def _fixture(root, *, alias=False):
    root.mkdir()
    names = list(LABELS[:-1])
    if alias:
        names[0] = "Ace Clubs"
    (root / "data.yaml").write_text(yaml.safe_dump({
        "nc": 52, "names": names,
        "roboflow": {"workspace": "joshuas-workspace", "project": "playing-cards-9gfac", "version": 2},
    }))
    paths = []
    for number, split in enumerate(("train", "valid")):
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        name = f"frame_jpg.rf.{number:04x}"
        image = images / f"{name}.png"
        Image.new("RGB", (16, 16), (number * 40, 10, 20)).save(image)
        (labels / f"{name}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        paths.append(image.relative_to(root).as_posix())
    return paths


def _review(path, groups, source, *, class_map_path=None):
    proposal, _, _ = face.prepare_face_source(
        source, path.parent / "unused", dry_run=True, class_map_path=class_map_path)
    path.write_text(json.dumps({
        "schema_version": 1, "source_id": "playing-cards-seed",
        "artifact_fingerprint": proposal["group_proposal"]["artifact_fingerprint"],
        "review": {"reviewer": "Synthetic fixture", "review_date": "2026-10-04",
                   "method": "Original scene provenance verified in test fixture",
                   "approved": True, "similarity_candidates_reviewed": True},
        "groups": groups,
    }, sort_keys=True))


def test_dry_run_proposes_groups_without_writing_or_approving(tmp_path):
    root = tmp_path / "face"
    paths = _fixture(root)
    output = tmp_path / "workspace-face"
    report, records, actual_root = face.prepare_face_source(root, output, dry_run=True)
    assert report["artifact_audit"]["artifact_status"] == "SOURCE_ARTIFACT_VALID"
    assert report["status"] == "NOT READY"
    assert set(report["group_proposal"]["groups"]) == set(paths)
    assert records is None and actual_root is None
    assert not output.exists()


def test_review_cannot_split_two_generated_variants(tmp_path):
    root = tmp_path / "face"
    paths = _fixture(root)
    reviewed = tmp_path / "groups.json"
    _review(reviewed, {paths[0]: "scene-a", paths[1]: "scene-b"}, root)
    report, records, actual_root = face.prepare_face_source(
        root, tmp_path / "workspace-face", reviewed,
    )
    assert report["status"] == "NOT READY"
    assert any("family" in reason.lower() for reason in report["reasons"])
    assert records is None and actual_root is None


def test_reviewed_alias_import_preserves_one_family_and_original_pixels(tmp_path):
    root = tmp_path / "face"
    paths = _fixture(root, alias=True)
    reviewed = tmp_path / "groups.json"
    aliases = tmp_path / "aliases.json"
    aliases.write_text('{"Ace Clubs": "AC"}')
    _review(reviewed, {path: "original-scene" for path in paths}, root, class_map_path=aliases)
    report, records, actual_root = face.prepare_face_source(
        root, tmp_path / "workspace-face", reviewed, class_map_path=aliases,
    )
    assert report["status"] == "READY"
    assert report["reviewed_groups_sha256"] == hashlib.sha256(reviewed.read_bytes()).hexdigest()
    assert report["artifact_audit"]["class_index_to_label"]["0"] == "AC"
    assert {record.scene_id for record in records} == {"original-scene"}
    assert {obj.label for record in records for obj in record.objects} == {"AC"}
    assert actual_root == root.resolve()


def test_archive_rejects_parent_escape_before_extracting(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "bad")
    report, records, actual_root = face.prepare_face_source(
        archive, tmp_path / "workspace-face", dry_run=True,
    )
    assert report["status"] == "NOT READY"
    assert records is None and actual_root is None
    assert not (tmp_path / "escape.txt").exists()


def test_original_zip_extracts_only_after_review_and_keeps_archive_hash(tmp_path):
    source = tmp_path / "original"
    paths = _fixture(source)
    reviewed = tmp_path / "groups.json"
    archive = tmp_path / "joshua-v2.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                output.write(path, (Path("export") / path.relative_to(source)).as_posix())
    _review(reviewed, {path: "one-original-scene" for path in paths}, archive)
    destination = tmp_path / "durable"
    report, records, actual_root = face.prepare_face_source(
        archive, destination, reviewed,
    )
    assert report["status"] == "READY", report["reasons"]
    assert report["artifact_audit"]["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert len(records) == 2
    assert actual_root == (destination / "export").resolve()


def test_review_fingerprint_rejects_changed_image_bytes(tmp_path):
    root = tmp_path / "face"
    paths = _fixture(root)
    reviewed = tmp_path / "groups.json"
    _review(reviewed, {path: "one-scene" for path in paths}, root)
    Image.new("RGB", (16, 16), "blue").save(root / paths[0])
    report, _, _ = face.prepare_face_source(root, tmp_path / "destination", reviewed, dry_run=True)
    assert report["status"] == "NOT READY"
    assert any("fingerprint" in reason for reason in report["reasons"])


def test_archive_rejects_normalized_duplicate_paths(tmp_path):
    archive = tmp_path / "colliding.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("data.yaml", "first")
        output.writestr("./data.yaml", "second")
    report, _, _ = face.prepare_face_source(archive, tmp_path / "destination", dry_run=True)
    assert report["status"] == "NOT READY"
    assert any("Unsafe" in reason for reason in report["reasons"])
