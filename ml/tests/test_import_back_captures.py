"""Real small raster fixtures for the CardViper-owned BACK capture boundary."""
import hashlib
import json

from PIL import Image
import pytest


def capture(root, name="a.png", *, color="red", session="s1", design="blue-weave",
            mode="tight-back-crop", boxes=None):
    path = root / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 12), color).save(path)
    row = {"capture_date": "2026-10-03", "device": "owned-camera", "session_id": session,
           "scene_id": f"{session}-{name}", "deck_design": design,
           "image_path": f"images/{name}",
           "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "annotation_provenance": "hand labeled by owner", "capture_mode": mode}
    if mode == "tight-back-crop":
        row["full_image_back"] = True
    else:
        row["boxes"] = boxes if boxes is not None else [
            {"annotation_id": "back-1", "label": "BACK", "bbox": [2, 1, 18, 11]}]
    return row


def manifest(root, rows, **changes):
    data = {"schema_version": 1, "source_id": "cardviper-back", "owner": "CardViper team",
            "rights": {"classifier": True, "detector": False,
                       "evidence": "owner captured and licensed for CardViper use"},
            "captures": rows}
    data.update(changes)
    (root / "captures.json").write_text(json.dumps(data), encoding="utf-8")


def test_tight_crop_import_preserves_source_bytes_hash_and_session(tmp_path):
    from cardviper_ml.import_back_captures import import_back_source
    row = capture(tmp_path)
    manifest(tmp_path, [row])
    before = (tmp_path / row["image_path"]).read_bytes()
    records = import_back_source(tmp_path)
    assert len(records) == 1
    assert records[0].source_id == "cardviper-back"
    assert records[0].session_id == "s1"
    assert records[0].scene_id == "s1-a.png"
    assert records[0].objects[0].label == "BACK"
    assert records[0].objects[0].bbox == (0, 0, 20, 12)
    assert records[0].image_sha256 == row["image_sha256"]
    assert (tmp_path / row["image_path"]).read_bytes() == before


def test_scene_requires_real_boxes_and_keeps_them(tmp_path):
    from cardviper_ml.import_back_captures import import_back_source
    row = capture(tmp_path, mode="annotated-scene")
    manifest(tmp_path, [row])
    assert import_back_source(tmp_path)[0].objects[0].bbox == (2, 1, 18, 11)
    row["boxes"] = []
    manifest(tmp_path, [row])
    with pytest.raises(ValueError, match="box"):
        import_back_source(tmp_path)


@pytest.mark.parametrize("change, error", [
    ({"full_image_back": False}, "full_image_back"),
    ({"image_sha256": "0" * 64}, "SHA"),
    ({"image_path": "../escape.png"}, "path"),
    ({"annotation_provenance": ""}, "annotation_provenance"),
])
def test_missing_or_false_capture_evidence_rejected(tmp_path, change, error):
    from cardviper_ml.import_back_captures import import_back_source
    row = capture(tmp_path)
    row.update(change)
    manifest(tmp_path, [row])
    with pytest.raises(ValueError, match=error):
        import_back_source(tmp_path)


def test_audit_reports_duplicate_clues_and_requires_human_review(tmp_path):
    from cardviper_ml.import_back_captures import audit_back_source
    rows = [capture(tmp_path, f"{i}.png", session=f"s{i}",
                    design="weave" if i < 2 else "star", color="red") for i in range(3)]
    manifest(tmp_path, rows)
    report = audit_back_source(tmp_path)
    assert report["sample_count"] == 3
    assert report["session_count"] == 3
    assert report["design_count"] == 2
    assert report["dimensions"] == {"20x12": 3}
    assert len(report["exact_duplicates"]) == 1
    assert report["split_eligibility"] == "NOT READY"
    assert report["status"] == "NOT READY"
    assert report["near_duplicate_clues"]


def test_audit_structural_readiness_still_needs_explicit_review(tmp_path):
    from cardviper_ml.import_back_captures import audit_back_source
    rows = [capture(tmp_path, f"{i}.png", session=f"s{i}",
                    design=f"design-{i % 2}", color=color)
            for i, color in enumerate(("red", "green", "blue"))]
    manifest(tmp_path, rows)
    report = audit_back_source(tmp_path)
    assert report["split_eligibility"] == "READY"
    assert report["status"] == "NOT READY"
    manifest(tmp_path, rows, review={"reviewer": "owner", "notes": "Three independently captured sessions; inspected designs and similarity clues.",
                                     "approved": True, "session_independence_confirmed": True,
                                     "near_duplicate_clues_reviewed": True})
    report = audit_back_source(tmp_path)
    assert report["status"] == "READY"


def test_audit_rejects_rights_without_evidence(tmp_path):
    from cardviper_ml.import_back_captures import import_back_source
    manifest(tmp_path, [capture(tmp_path)], rights={"classifier": True, "detector": True, "evidence": ""})
    with pytest.raises(ValueError, match="rights"):
        import_back_source(tmp_path)


def test_classifier_readiness_does_not_grant_detector_permission(tmp_path):
    from cardviper_ml.import_back_captures import audit_back_source
    rows = [capture(tmp_path, f"{i}.png", session=f"s{i}",
                    design=f"design-{i % 2}", color=color)
            for i, color in enumerate(("red", "green", "blue"))]
    manifest(tmp_path, rows,
             rights={"classifier": True, "detector": False,
                     "evidence": "owner authorized classifier only"},
             review={"reviewer": "owner", "notes": "Independent sessions and clues reviewed",
                     "approved": True, "session_independence_confirmed": True,
                     "near_duplicate_clues_reviewed": True})
    report = audit_back_source(tmp_path)
    assert report["status"] == "READY"
    assert report["rights"]["detector"] is False


def test_detector_permission_needs_separate_evidence(tmp_path):
    from cardviper_ml.import_back_captures import import_back_source
    row = capture(tmp_path)
    rights = {"classifier": True, "detector": True,
              "evidence": "owner authorized classifier use"}
    manifest(tmp_path, [row], rights=rights)
    with pytest.raises(ValueError, match="detector_evidence"):
        import_back_source(tmp_path)
    rights["detector_evidence"] = "owner separately authorized detector use"
    manifest(tmp_path, [row], rights=rights)
    assert len(import_back_source(tmp_path)) == 1
