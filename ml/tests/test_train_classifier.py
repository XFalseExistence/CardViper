import json

import pytest

from cardviper_ml.train_classifier import require_ready, validate_crop_coverage


def test_real_training_gate_has_no_override(tmp_path):
    report = tmp_path / "preflight.json"
    report.write_text(json.dumps({"status": "NOT READY", "ready": False}))
    with pytest.raises(ValueError, match="READY"):
        require_ready(report)


def test_injected_or_missing_crop_rejected():
    expected = {("owned", "a" * 64, "1", "AC")}
    with pytest.raises(ValueError, match="coverage"):
        validate_crop_coverage(expected, set())
    with pytest.raises(ValueError, match="coverage"):
        validate_crop_coverage(expected, {("owned", "b" * 64, "1", "AC")})
