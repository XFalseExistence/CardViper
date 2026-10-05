import hashlib

import pytest

from cardviper_ml.inspect_detector_litert import inspect_artifact


def test_inspector_reads_real_float_tensor_metadata(tmp_path):
    tf = pytest.importorskip("tensorflow")

    class Tiny(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 2], tf.float32, name="input")])
        def call(self, value):
            return {"scores": value + 1.0}

    model = Tiny()
    saved = tmp_path / "saved"
    tf.saved_model.save(model, str(saved), signatures=model.call)
    content = tf.lite.TFLiteConverter.from_saved_model(str(saved)).convert()
    artifact = tmp_path / "tiny.tflite"
    artifact.write_bytes(content)
    report = inspect_artifact(artifact)
    assert report["artifact_sha256"] == hashlib.sha256(content).hexdigest()
    assert report["file_size_bytes"] == len(content)
    assert report["input_tensors"][0]["shape"] == [1, 2]
    assert report["input_tensors"][0]["dtype"] == "float32"
    assert report["output_tensors"][0]["dtype"] == "float32"
    assert report["observed_outputs"][0]["minimum"] == pytest.approx(1.0)
    assert report["output_semantics"] == "UNKNOWN"
    assert report["signature_to_raw"]["serving_default"]["scores"] == [
        report["output_tensors"][0]["name"]]


def test_inspector_rejects_non_tflite_bytes(tmp_path):
    artifact = tmp_path / "bad.tflite"
    artifact.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="TFLite"):
        inspect_artifact(artifact)
