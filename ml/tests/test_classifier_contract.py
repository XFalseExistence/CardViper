from pathlib import Path

import pytest

from cardviper_ml.classifier_contract import build_contract


def test_contract_requires_real_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_contract(tmp_path / "missing.tflite", {}, tmp_path / "labels.txt", "eval.json")
