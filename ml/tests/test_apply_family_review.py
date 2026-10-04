import json

import pytest

from cardviper_ml.apply_family_review import apply_review
from cardviper_ml.build_family_review import build_review_pack
from test_build_family_review import review_fixture


def test_review_requires_complete_explicit_decisions_and_rejects_stale_source(tmp_path):
    args = review_fixture(tmp_path)
    pack = tmp_path / "pack"
    build_review_pack(*args, pack)
    draft = pack / "review.json"
    output = tmp_path / "groups.json"
    result = apply_review(*args, draft, output)
    assert result["status"] == "NOT READY"
    assert result["unresolved_candidate_count"] == 1
    data = json.loads(draft.read_text())
    data["decisions"] = []
    bad = tmp_path / "missing.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exactly"):
        apply_review(*args, bad, tmp_path / "missing-groups.json")
    data = json.loads(draft.read_text())
    data["archive_sha256"] = "0" * 64
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="stale|changed"):
        apply_review(*args, bad, tmp_path / "stale-groups.json")


@pytest.mark.parametrize("decision,expected_groups", [("same_family", 1), ("independent", 2)])
def test_human_review_applies_only_explicit_decision(tmp_path, decision, expected_groups):
    args = review_fixture(tmp_path)
    pack = tmp_path / "pack"
    build_review_pack(*args, pack)
    data = json.loads((pack / "review.json").read_text())
    data.update({"reviewer": "human", "review_date": "2026-10-04",
                 "method": "Compared source images and capture provenance", "approved": True})
    data["decisions"][0].update({"decision": decision, "notes": "Inspected both originals"})
    decision_file = tmp_path / "decisions.json"
    decision_file.write_text(json.dumps(data))
    output = tmp_path / "groups.json"
    result = apply_review(*args, decision_file, output)
    assert result["status"] == "READY"
    groups = json.loads(output.read_text())
    assert len(set(groups["groups"].values())) == expected_groups
    assert groups["review"]["approved"] is True
