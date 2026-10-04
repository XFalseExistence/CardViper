from cardviper_ml.classifier_model import ModelConfig, model_spec


def test_model_spec_freezes_53_outputs_and_two_learning_rates():
    spec = model_spec(ModelConfig())
    assert spec["architecture"] == "MobileNetV3Small"
    assert spec["output_width"] == 53
    assert spec["preprocessing"] == "MobileNetV3Small include_preprocessing=True; RGB float32 0..255"
    assert spec["fine_tune_learning_rate"] < spec["head_learning_rate"]
