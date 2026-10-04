from cardviper_ml.export_classifier import write_labels
from cardviper_ml.export_classifier import validate_export_evidence
from cardviper_ml.labels import LABELS
import hashlib
import json

import pytest


def test_canonical_labels_file_is_exact(tmp_path):
    output = tmp_path / "labels.txt"
    write_labels(output)
    assert output.read_text().splitlines() == list(LABELS)
    assert output.read_text().splitlines()[52] == "BACK"


def test_export_rejects_test_report_for_a_different_checkpoint(tmp_path):
    model = tmp_path / "chosen.keras"
    model.write_bytes(b"chosen")
    report = tmp_path / "test.json"
    report.write_text(json.dumps({"evaluated_split": "test", "model_sha256": "0" * 64,
                                  "test_split_manifest_sha256": "a" * 64,
                                  "test_crop_manifest_sha256": "b" * 64}))
    metadata = {"preflight_status": "READY", "selected_checkpoint": str(model),
                "selected_checkpoint_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "split_manifest_sha256": {"test": "a" * 64}, "evaluation_report": str(report)}
    with pytest.raises(ValueError, match="checkpoint"):
        validate_export_evidence(metadata, model)
