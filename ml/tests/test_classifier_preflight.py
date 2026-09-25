import hashlib
import json
import subprocess
import sys
from dataclasses import replace

from cardviper_ml.labels import LABELS
from cardviper_ml.manifest import Annotation, ImageRecord, write_manifest
from cardviper_ml.splits import SPLIT_NAMES, write_splits


def _record(split, index, label, *, held_out=False):
    identity = f"{split}-{index}-{label}"
    return ImageRecord(
        source_id="owned",
        scene_id=f"scene-{identity}",
        session_id=f"session-{identity}",
        image_path=f"images/{identity}.png",
        width=20,
        height=20,
        image_sha256=hashlib.sha256(identity.encode()).hexdigest(),
        annotation_path=f"labels/{identity}.txt",
        objects=(Annotation("1", label, (1, 1, 10, 15)),),
        held_out=held_out,
    )


def _fixture(tmp_path):
    splits = {
        name: [_record(name, index, label) for index, label in enumerate(LABELS)]
        for name in ("train", "val", "test")
    }
    splits["holdout"] = [_record("holdout", 0, "BACK", held_out=True)]
    split_dir = tmp_path / "splits"
    write_splits(split_dir, splits)
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({
        "schema_version": 1,
        "sources": [{
            "source_id": "owned",
            "name": "Owned fixture",
            "url": "https://example.test/cardviper-owned",
            "version": "capture-1",
            "license": "Proprietary-CardViper",
            "attribution": "CardViper test fixture",
            "verified": True,
            "local_root": "CARDVIPER_OWNED_ROOT",
            "permitted_use": {"detector": True, "classifier": True},
            "notes": "Synthetic test metadata only."
        }]
    }))
    return sources, split_dir, splits


def _rewrite(split_dir, splits):
    for name in SPLIT_NAMES:
        write_manifest(split_dir / f"{name}.jsonl", splits[name])


def test_ready_summary_reports_counts_groups_distribution_and_sources(tmp_path):
    from cardviper_ml.classifier_preflight import audit_classifier_data

    sources, split_dir, _ = _fixture(tmp_path)
    report = audit_classifier_data(sources, split_dir)

    assert report["status"] == "READY"
    assert report["ready"] is True
    assert report["reasons"] == []
    assert report["total_images"] == 160
    assert report["total_card_annotations"] == 160
    assert report["independent_groups"] == 160
    assert report["split_image_counts"] == {"train": 53, "val": 53, "test": 53, "holdout": 1}
    assert report["split_group_counts"] == {"train": 53, "val": 53, "test": 53, "holdout": 1}
    assert report["per_class_annotation_counts"]["BACK"] == {
        "train": 1, "val": 1, "test": 1, "holdout": 1, "total": 4,
    }
    assert report["back_count"] == 4
    assert report["examples_per_class"] == {"min": 3, "median": 3, "max": 4}
    assert report["permitted_source_ids"] == ["owned"]
    assert report["held_out_count"] == 1


def test_unverified_or_unpermitted_source_fails_closed(tmp_path):
    from cardviper_ml.classifier_preflight import audit_classifier_data

    sources, split_dir, _ = _fixture(tmp_path)
    data = json.loads(sources.read_text())
    data["sources"][0].update(verified=False, url=None, license=None, attribution=None)
    data["sources"][0]["permitted_use"] = {"detector": False, "classifier": False}
    sources.write_text(json.dumps(data))

    report = audit_classifier_data(sources, split_dir)
    assert report["status"] == "NOT READY"
    assert {reason["code"] for reason in report["reasons"]} == {"SOURCE_NOT_PERMITTED"}


def test_every_class_must_exist_in_train_validation_and_test(tmp_path):
    from cardviper_ml.classifier_preflight import audit_classifier_data

    sources, split_dir, splits = _fixture(tmp_path)
    splits["train"] = [row for row in splits["train"] if row.objects[0].label != "BACK"]
    splits["val"] = [row for row in splits["val"] if row.objects[0].label != "AC"]
    splits["test"] = [row for row in splits["test"] if row.objects[0].label != "2C"]
    _rewrite(split_dir, splits)

    report = audit_classifier_data(sources, split_dir)
    reasons = {reason["code"]: reason for reason in report["reasons"]}
    assert reasons["ZERO_TRAIN_CLASS"]["labels"] == ["BACK"]
    assert reasons["VALIDATION_CLASS_MISSING"]["labels"] == ["AC"]
    assert reasons["TEST_CLASS_MISSING"]["labels"] == ["2C"]


def test_label_absent_everywhere_is_reported_separately(tmp_path):
    from cardviper_ml.classifier_preflight import audit_classifier_data

    sources, split_dir, splits = _fixture(tmp_path)
    for name in SPLIT_NAMES:
        splits[name] = [row for row in splits[name] if row.objects[0].label != "KS"]
    _rewrite(split_dir, splits)

    report = audit_classifier_data(sources, split_dir)
    reason = next(reason for reason in report["reasons"] if reason["code"] == "REQUIRED_LABEL_MISSING")
    assert reason["labels"] == ["KS"]


def test_leakage_pixel_holdout_and_duplicate_hashes_fail_closed(tmp_path):
    from cardviper_ml.classifier_preflight import audit_classifier_data

    for corruption, expected_text in (
        ("leak", "group"), ("pixel", "held-out"), ("duplicate", "Duplicate")
    ):
        case = tmp_path / corruption
        case.mkdir()
        sources, split_dir, splits = _fixture(case)
        if corruption == "leak":
            splits["val"][0] = replace(splits["val"][0], scene_id=splits["train"][0].scene_id)
        elif corruption == "pixel":
            splits["train"][0] = replace(splits["train"][0], held_out=True)
        else:
            splits["val"][0] = replace(splits["val"][0], image_sha256=splits["train"][0].image_sha256)
        _rewrite(split_dir, splits)
        report = audit_classifier_data(sources, split_dir)
        assert report["status"] == "NOT READY"
        assert report["reasons"][0]["code"] == "SPLIT_INVALID"
        assert expected_text.lower() in report["reasons"][0]["message"].lower()


def test_unknown_label_and_malformed_manifest_fail_closed(tmp_path):
    from cardviper_ml.classifier_preflight import audit_classifier_data

    for replacement in ('"label": "JOKER"', '"width": "twenty"'):
        case = tmp_path / replacement.split()[-1].strip('"')
        case.mkdir()
        sources, split_dir, _ = _fixture(case)
        path = split_dir / "train.jsonl"
        text = path.read_text()
        text = text.replace('"label": "AC"', replacement, 1) if "label" in replacement else text.replace('"width": 20', replacement, 1)
        path.write_text(text)
        report = audit_classifier_data(sources, split_dir)
        assert report["status"] == "NOT READY"
        assert report["reasons"][0]["code"] == "SPLIT_INVALID"


def test_cli_writes_json_and_markdown_and_uses_exit_status(tmp_path):
    sources, split_dir, _ = _fixture(tmp_path)
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    ml_root = str(__file__).rsplit("/tests/", 1)[0]
    result = subprocess.run([
        sys.executable, "-m", "cardviper_ml.classifier_preflight",
        "--sources", str(sources), "--splits", str(split_dir),
        "--json", str(json_path), "--markdown", str(markdown_path),
    ], capture_output=True, text=True, cwd=ml_root)
    assert result.returncode == 0, result.stderr
    assert json.loads(json_path.read_text())["status"] == "READY"
    markdown = markdown_path.read_text()
    assert "# CardViper classifier data preflight" in markdown
    assert "**READY**" in markdown
    assert "BACK" in markdown

    data = json.loads(sources.read_text())
    data["sources"][0]["permitted_use"]["classifier"] = False
    sources.write_text(json.dumps(data))
    result = subprocess.run([
        sys.executable, "-m", "cardviper_ml.classifier_preflight",
        "--sources", str(sources), "--splits", str(split_dir),
    ], capture_output=True, text=True, cwd=ml_root)
    assert result.returncode == 2
    assert "NOT READY" in result.stdout
