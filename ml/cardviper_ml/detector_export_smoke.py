"""Export a random-weight, one-class OD API FPNLite 320 model; never train or publish it."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def _tensor(detail):
    import numpy as np
    return {
        "name": detail["name"],
        "index": int(detail["index"]),
        "shape": [int(value) for value in detail["shape"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "quantization": [float(detail["quantization"][0]), int(detail["quantization"][1])],
    }


def run_smoke(odapi_root, output, *, seed=42):
    root = Path(odapi_root).resolve()
    config_path = root / "research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config"
    exporter_path = root / "research/object_detection/export_tflite_graph_lib_tf2.py"
    if not config_path.is_file() or not exporter_path.is_file():
        raise ValueError("Expected pinned TensorFlow Object Detection API source with FPNLite 320 config")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    (output / "matplotlib").mkdir()
    sys.path.insert(0, str(root / "research"))
    sys.path.insert(0, str(root / "research/slim"))
    report = {"status": "FAIL", "semantic_classes": ["CARD"], "input_size": [320, 320],
              "seed": seed, "stages": [], "failed_stage": "imports"}
    stage = "imports"
    try:
        import numpy as np
        import tensorflow as tf
        from ai_edge_litert.interpreter import Interpreter
        from google.protobuf import text_format
        from object_detection.builders import model_builder
        from object_detection import export_tflite_graph_lib_tf2
        from object_detection.protos import pipeline_pb2

        tf.random.set_seed(seed)
        np.random.seed(seed)
        report["python_version"] = sys.version.split()[0]
        report["tensorflow_version"] = tf.__version__
        report["odapi_source_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        config = pipeline_pb2.TrainEvalPipelineConfig()
        text_format.Merge(config_path.read_text(encoding="utf-8"), config)
        config.model.ssd.num_classes = 1
        (output / "pipeline.config").write_text(text_format.MessageToString(config), encoding="utf-8")

        stage = "instantiate"
        model = model_builder.build(config.model, is_training=False)
        report["stages"].append(stage)
        print("INSTANTIATE_OK", flush=True)

        stage = "checkpoint"
        image, shapes = model.preprocess(tf.zeros([1, 320, 320, 3], tf.float32))
        model.predict(image, true_image_shapes=shapes)
        checkpoint_dir = output / "checkpoint"
        checkpoint_dir.mkdir()
        tf.train.Checkpoint(model=model).save(str(checkpoint_dir / "ckpt"))
        report["stages"].append(stage)
        print("CHECKPOINT_OK", flush=True)

        stage = "export SavedModel"
        export_tflite_graph_lib_tf2.export_tflite_model(
            config, str(checkpoint_dir), str(output / "export"),
            max_detections=10, use_regular_nms=False)
        report["stages"].append(stage)
        print("SAVEDMODEL_OK", flush=True)

        stage = "TFLite conversion"
        converter = tf.lite.TFLiteConverter.from_saved_model(str(output / "export/saved_model"))
        artifact = converter.convert()
        model_path = output / "detector-smoke.tflite"
        model_path.write_bytes(artifact)
        report["artifact_size_bytes"] = len(artifact)
        report["artifact_sha256"] = hashlib.sha256(artifact).hexdigest()
        report["stages"].append(stage)
        print("TFLITE_CONVERT_OK", len(artifact), flush=True)

        stage = "LiteRT loading"
        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        report["input_tensors"] = [_tensor(detail) for detail in interpreter.get_input_details()]
        report["output_tensors"] = [_tensor(detail) for detail in interpreter.get_output_details()]
        report["stages"].append(stage)
        print("LITERT_LOAD_OK", flush=True)

        stage = "inference"
        detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"], dtype=detail["dtype"]))
        interpreter.invoke()
        report["observed_outputs"] = [
            {"name": detail["name"], "minimum": float(interpreter.get_tensor(detail["index"]).min()),
             "maximum": float(interpreter.get_tensor(detail["index"]).max())}
            for detail in interpreter.get_output_details()]
        report["stages"].append(stage)
        report["status"] = "PASS"
        report["failed_stage"] = None
        print("INFERENCE_OK", flush=True)
    except Exception as error:
        report["failed_stage"] = stage
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
        raise
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odapi-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_smoke(args.odapi_root, args.output, seed=args.seed)
    print(json.dumps({key: report[key] for key in
                      ("status", "artifact_size_bytes", "artifact_sha256", "stages")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
