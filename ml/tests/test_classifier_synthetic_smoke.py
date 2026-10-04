"""Optional integration smoke: synthetic output is never a performance result."""

import json

import pytest
from PIL import Image


def test_synthetic_model_fit_export_inspection_and_contract(tmp_path):
    tf = pytest.importorskip("tensorflow")
    np = pytest.importorskip("numpy")
    from cardviper_ml.classifier_model import ModelConfig, build_model, compile_stage, model_spec
    from cardviper_ml.inspect_litert import inspect_tflite
    from cardviper_ml.classifier_contract import build_contract
    from cardviper_ml.export_classifier import export_model
    from cardviper_ml.classifier_dataset import load_crop_split, rgb_array
    from cardviper_ml.classifier_augmentation import augment_image
    from cardviper_ml.eval_classifier import evaluate_scores

    config = ModelConfig(input_size=96)
    crop_root = tmp_path / "train"
    crop_root.mkdir()
    crop_rows = []
    for index, label in enumerate(("AC", "BACK")):
        Image.new("RGB", (96, 96), (index * 60, 20, 30)).save(crop_root / f"{index}.png")
        crop_rows.append({"schema_version": 1, "source_id": "SYNTHETIC-SMOKE-ONLY",
                          "scene_id": f"synthetic-{index}", "session_id": None,
                          "held_out": False, "source_image": f"source/{index}.png",
                          "source_image_sha256": f"{index:064x}",
                          "source_annotation": f"ann/{index}.txt", "annotation_id": "1",
                          "source_bbox": [0, 0, 96, 96], "label": label,
                          "crop_bbox": [0, 0, 96, 96], "padding": 0,
                          "crop_path": f"{index}.png"})
    (crop_root / "crops.jsonl").write_text("".join(json.dumps(row) + "\n" for row in crop_rows))
    samples = load_crop_split(crop_root, "train")
    model, backbone = build_model(config, weights=None)
    pixels = np.asarray([np.asarray(augment_image(rgb_array(sample, 96), split="train", seed=index), dtype=np.float32)
                         for index, sample in enumerate(samples)])
    labels = np.asarray([sample.label_index for sample in samples], dtype=np.int32)
    model.fit(pixels, labels, batch_size=2, epochs=1, verbose=0)
    compile_stage(model, backbone, config, fine_tune=True)
    model.fit(pixels, labels, batch_size=2, epochs=1, verbose=0)
    scores = model.predict(pixels, verbose=0).tolist()
    synthetic_evaluation = evaluate_scores([sample.label for sample in samples], scores, split="train")
    assert synthetic_evaluation["sample_count"] == 2
    keras_path = tmp_path / "synthetic.keras"
    model.save(keras_path)
    metadata = {**model_spec(config), "evaluation_report": "SYNTHETIC-SMOKE-ONLY"}
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    path = export_model(keras_path, tmp_path / "export", metadata_path, smoke=True)
    inspected = inspect_tflite(path)
    assert inspected["output_tensors"][0]["shape"][-1] == 53
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_tensor = interpreter.get_input_details()[0]
    output_tensor = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_tensor["index"], pixels[:1])
    interpreter.invoke()
    lite_scores = interpreter.get_tensor(output_tensor["index"])
    np.testing.assert_allclose(lite_scores, np.asarray(scores[:1]), rtol=1e-3, atol=1e-3)
    contract = build_contract(path, model_spec(config), path.parent / "labels.txt", "SYNTHETIC-SMOKE-ONLY", smoke=True)
    assert contract["back_index"] == 52
    assert contract["model_sha256"] == inspected["model_sha256"]
    assert contract["index_to_label"]["0"] == "AC"
    assert contract["index_to_label"]["52"] == "BACK"
