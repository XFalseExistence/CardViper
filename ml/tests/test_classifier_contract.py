from pathlib import Path

import pytest

from cardviper_ml.classifier_contract import build_contract
from cardviper_ml.classifier_model import ModelConfig, model_spec
from cardviper_ml.export_classifier import write_labels


def test_contract_requires_real_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_contract(tmp_path / "missing.tflite", {}, tmp_path / "labels.txt", "eval.json")


def test_standalone_contract_rejects_unverified_training_before_inspection(tmp_path):
    model = tmp_path / "fake.tflite"
    model.write_bytes(b"not a model")
    labels = write_labels(tmp_path / "labels.txt")
    with pytest.raises(ValueError, match="Evaluation report"):
        build_contract(model, model_spec(ModelConfig()), labels, "arbitrary-report.json")
