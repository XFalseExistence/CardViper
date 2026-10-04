"""Verify the selected checkpoint and held-out TEST report before export."""

import hashlib
import json
from pathlib import Path
import re


def validate_export_evidence(metadata, model_path):
    model_path = Path(model_path)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if (metadata.get("preflight_status") != "READY" or
            Path(metadata.get("selected_checkpoint", "")).resolve() != model_path.resolve() or
            metadata.get("selected_checkpoint_sha256") != model_sha):
        raise ValueError("Real export requires the selected READY checkpoint and matching bytes")
    report_path = Path(metadata.get("evaluation_report") or "")
    if not report_path.is_file():
        raise ValueError("Real export requires a TEST evaluation report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("evaluated_split") != "test" or report.get("model_sha256") != model_sha or
            report.get("test_split_manifest_sha256") != metadata.get("split_manifest_sha256", {}).get("test") or
            not isinstance(report.get("sample_count"), int) or report["sample_count"] <= 0 or
            not isinstance(report.get("test_crop_manifest_sha256"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", report["test_crop_manifest_sha256"])):
        raise ValueError("TEST report does not match selected checkpoint and split provenance")
    return report
