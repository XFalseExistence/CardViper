import json
import os
from pathlib import Path
import subprocess

import pytest

from cardviper_ml.detector_export_smoke import run_smoke


def test_smoke_requires_pinned_odapi_source(tmp_path):
    with pytest.raises(ValueError, match="Object Detection API"):
        run_smoke(tmp_path / "missing", tmp_path / "output")


@pytest.mark.skipif(not os.getenv("CARDVIPER_DETECTOR_PYTHON") or
                    not os.getenv("CARDVIPER_ODAPI_ROOT"),
                    reason="isolated OD API smoke environment not supplied")
def test_real_one_class_export_and_litert_inference(tmp_path):
    python = os.environ["CARDVIPER_DETECTOR_PYTHON"]
    root = os.environ["CARDVIPER_ODAPI_ROOT"]
    output = tmp_path / "smoke"
    process = subprocess.run([python, "-m", "cardviper_ml.detector_export_smoke",
                              "--odapi-root", root, "--output", str(output)],
                             capture_output=True, text=True, env=os.environ.copy(),
                             timeout=900, check=False)
    assert process.returncode == 0, process.stdout[-2000:] + process.stderr[-3000:]
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "PASS"
    assert report["semantic_classes"] == ["CARD"]
    assert report["input_size"] == [320, 320]
    assert report["stages"] == ["instantiate", "checkpoint", "export SavedModel",
                                "TFLite conversion", "LiteRT loading", "inference"]
    assert (output / "detector-smoke.tflite").is_file()
