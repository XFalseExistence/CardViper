"""Train a one-class CARD detector for the local-only Session A phone demo."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .labels import LABELS
from .manifest import ImageRecord, local_path, read_manifest


DEMO_USAGE = "SESSION_A_DEMO_ONLY"
INDEPENDENCE = "NOT_INDEPENDENT"
CARD = "CARD"
FPNLITE_CONFIG = "research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detector_pipeline_settings(*, batch_size: int, train_steps: int) -> dict:
    if any(type(value) is not int or value <= 0 for value in (batch_size, train_steps)):
        raise ValueError("batch_size and train_steps must be positive integers")
    return {
        "config_relative_path": FPNLITE_CONFIG,
        "class_names": [CARD],
        "num_classes": 1,
        "input_size": [320, 320],
        "batch_size": batch_size,
        "train_steps": train_steps,
    }


def detector_example(row: ImageRecord) -> dict:
    """Make framework-neutral CARD-only detector features from an image record."""
    if not isinstance(row, ImageRecord):
        raise ValueError("Expected a validated ImageRecord")
    labels = [obj.label for obj in row.objects]
    if any(label not in LABELS[:-1] for label in labels):
        raise ValueError("Session A detector accepts canonical face-up source labels only")
    return {
        "image_path": row.image_path,
        "image_sha256": row.image_sha256,
        "width": row.width,
        "height": row.height,
        "xmins": [obj.bbox[0] / row.width for obj in row.objects],
        "xmaxs": [obj.bbox[2] / row.width for obj in row.objects],
        "ymins": [obj.bbox[1] / row.height for obj in row.objects],
        "ymaxs": [obj.bbox[3] / row.height for obj in row.objects],
        "class_ids": [0 for _ in row.objects],
        "class_names": [CARD for _ in row.objects],
    }


def _tf_example(row: ImageRecord, source_root: Path):
    import tensorflow as tf

    value = detector_example(row)
    image_path = local_path(source_root, row.image_path)
    image_bytes = image_path.read_bytes()
    if hashlib.sha256(image_bytes).hexdigest() != row.image_sha256:
        raise ValueError(f"Source image hash changed: {row.image_path}")

    def ints(items):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=items))

    def floats(items):
        return tf.train.Feature(float_list=tf.train.FloatList(value=items))

    def bytes_list(items):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=items))

    features = {
        "image/height": ints([value["height"]]),
        "image/width": ints([value["width"]]),
        "image/filename": bytes_list([value["image_path"].encode("utf-8")]),
        "image/source_id": bytes_list([value["image_sha256"].encode("ascii")]),
        "image/encoded": bytes_list([image_bytes]),
        "image/format": bytes_list([image_path.suffix.lstrip(".").lower().encode("ascii")]),
        "image/object/bbox/xmin": floats(value["xmins"]),
        "image/object/bbox/xmax": floats(value["xmaxs"]),
        "image/object/bbox/ymin": floats(value["ymins"]),
        "image/object/bbox/ymax": floats(value["ymaxs"]),
        # TF OD API label-map values are one-indexed; `detector_example` keeps
        # the application semantic class zero-indexed for its own contract.
        "image/object/class/label": ints([class_id + 1 for class_id in value["class_ids"]]),
        "image/object/class/text": bytes_list([name.encode("ascii") for name in value["class_names"]]),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_session_a_tfrecords(records, root: Path, *, source_root: Path) -> dict:
    """Write a fresh one-class TFRecord from verified local source images."""
    import tensorflow as tf

    rows = list(records)
    if not rows:
        raise ValueError("Cannot write an empty Session A detector TFRecord")
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "session_a_card.tfrecord"
    with tf.io.TFRecordWriter(str(output)) as writer:
        for row in rows:
            writer.write(_tf_example(row, Path(source_root)).SerializeToString())
    return {"path": str(output), "sha256": _sha256(output), "image_count": len(rows),
            "semantic_classes": [CARD]}


def _read_demo_input(data_root: Path) -> tuple[dict, Path, list, list]:
    try:
        metadata = json.loads((data_root / "demo_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Missing or invalid Session A demo metadata") from error
    if (metadata.get("usage") != DEMO_USAGE or metadata.get("independence") != INDEPENDENCE or
            metadata.get("independent_evaluation") is not False or metadata.get("face_up_only") is not True or
            metadata.get("back_supported") is not False or metadata.get("classifier_labels") != list(LABELS[:-1])):
        raise ValueError("Session A demo metadata does not describe approved face-only detector input")
    source_root = Path(metadata.get("source_root", ""))
    if not source_root.is_dir():
        raise ValueError("Session A source root is unavailable")
    train = read_manifest(data_root / "train.jsonl")
    smoke = read_manifest(data_root / "smoke.jsonl")
    if not train or not smoke:
        raise ValueError("Session A detector train and smoke manifests must be nonempty")
    return metadata, source_root, train, smoke


def _require_odapi(root: Path) -> tuple[Path, Path, Path]:
    root = Path(root).resolve()
    config = root / FPNLITE_CONFIG
    model_main = root / "research/object_detection/model_main_tf2.py"
    exporter = root / "research/object_detection/export_tflite_graph_lib_tf2.py"
    if not config.is_file() or not model_main.is_file() or not exporter.is_file():
        raise ValueError("Expected pinned TensorFlow Object Detection API source with FPNLite 320 config")
    return root, config, model_main


def odapi_environment(odapi_root: Path, environment: dict[str, str] | None = None) -> dict[str, str]:
    """Supply the OD API's research and slim roots to its training subprocess."""
    result = dict(os.environ if environment is None else environment)
    research = Path(odapi_root) / "research"
    roots = [str(research), str(research / "slim")]
    if result.get("PYTHONPATH"):
        roots.append(result["PYTHONPATH"])
    result["PYTHONPATH"] = os.pathsep.join(roots)
    return result


def _configure_pipeline(odapi_root: Path, config_path: Path, *, base_checkpoint: Path,
                        label_map: Path, train_record: Path, smoke_record: Path,
                        batch_size: int):
    from google.protobuf import text_format
    from object_detection.protos import pipeline_pb2

    config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Merge(config_path.read_text(encoding="utf-8"), config)
    config.model.ssd.num_classes = 1
    config.train_config.batch_size = batch_size
    config.train_config.fine_tune_checkpoint = str(base_checkpoint)
    config.train_config.fine_tune_checkpoint_type = "detection"
    config.train_input_reader.label_map_path = str(label_map)
    del config.train_input_reader.tf_record_input_reader.input_path[:]
    config.train_input_reader.tf_record_input_reader.input_path.append(str(train_record))
    eval_reader = config.eval_input_reader[0]
    eval_reader.label_map_path = str(label_map)
    del eval_reader.tf_record_input_reader.input_path[:]
    eval_reader.tf_record_input_reader.input_path.append(str(smoke_record))
    return config


def train_session_a_demo_detector(data_root: Path, output: Path, *, odapi_root: Path,
                                  base_checkpoint: Path, seed: int, train_steps: int,
                                  batch_size: int) -> dict:
    """Train and convert a real, one-class Session A LiteRT detector."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    settings = detector_pipeline_settings(batch_size=batch_size, train_steps=train_steps)
    data_root = Path(data_root)
    source, source_root, train, smoke = _read_demo_input(data_root)
    odapi_root, config_path, model_main = _require_odapi(Path(odapi_root))
    base_checkpoint = Path(base_checkpoint)
    if not base_checkpoint.is_file() and not base_checkpoint.with_suffix(".index").is_file():
        raise ValueError("A local OD API-compatible base checkpoint is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
        (output / "matplotlib").mkdir()
        sys.path.insert(0, str(odapi_root / "research"))
        sys.path.insert(0, str(odapi_root / "research/slim"))
        from google.protobuf import text_format
        import tensorflow as tf
        from object_detection import export_tflite_graph_lib_tf2
        from object_detection.protos import pipeline_pb2
        from .inspect_detector_litert import inspect_artifact

        tf.random.set_seed(seed)
        records_dir = output / "tfrecords"
        train_record = write_session_a_tfrecords(train, records_dir / "train", source_root=source_root)
        smoke_record = write_session_a_tfrecords(smoke, records_dir / "smoke", source_root=source_root)
        label_map = output / "card_label_map.pbtxt"
        label_map.write_text("item { id: 1 name: 'CARD' }\n", encoding="utf-8")
        pipeline = _configure_pipeline(
            odapi_root, config_path, base_checkpoint=base_checkpoint, label_map=label_map,
            train_record=Path(train_record["path"]), smoke_record=Path(smoke_record["path"]),
            batch_size=batch_size,
        )
        pipeline_path = output / "pipeline.config"
        pipeline_path.write_text(text_format.MessageToString(pipeline), encoding="utf-8")
        model_dir = output / "model"
        process = subprocess.run([
            sys.executable, str(model_main), "--pipeline_config_path", str(pipeline_path),
            "--model_dir", str(model_dir), "--num_train_steps", str(train_steps),
            "--sample_1_of_n_eval_examples", "1", "--alsologtostderr",
        ], check=False, capture_output=True, text=True,
            env=odapi_environment(odapi_root), timeout=7200)
        (output / "training.stdout.log").write_text(process.stdout, encoding="utf-8")
        (output / "training.stderr.log").write_text(process.stderr, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"OD API training failed with exit code {process.returncode}")
        checkpoint = tf.train.latest_checkpoint(str(model_dir))
        if checkpoint is None:
            raise RuntimeError("OD API training did not produce a checkpoint")
        export_dir = output / "export"
        export_tflite_graph_lib_tf2.export_tflite_model(
            pipeline, str(model_dir), str(export_dir), max_detections=20, use_regular_nms=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir / "saved_model"))
        detector = output / "session_a_card_detector.tflite"
        detector.write_bytes(converter.convert())
        inspection = inspect_artifact(detector)
        input_shapes = [tensor["shape"] for tensor in inspection["input_tensors"]]
        if not inspection["contains_detection_postprocess_custom_op"] or input_shapes != [[1, 320, 320, 3]]:
            raise ValueError("Exported detector is incompatible with the Session A detector contract builder")
        metadata = {
            "schema_version": 1,
            "usage": DEMO_USAGE,
            "independence": INDEPENDENCE,
            "independent_evaluation": False,
            "face_up_only": True,
            "back_supported": False,
            "semantic_classes": [CARD],
            "archive_sha256": source["archive_sha256"],
            "odapi_source_commit": subprocess.check_output(
                ["git", "-C", str(odapi_root), "rev-parse", "HEAD"], text=True).strip(),
            "settings": settings,
            "base_checkpoint": str(base_checkpoint),
            "selected_checkpoint": checkpoint,
            "train_record": train_record,
            "smoke_record": smoke_record,
            "detector_path": str(detector),
            "detector_sha256": _sha256(detector),
            "inspection": inspection,
            "smoke_note": "Session A deterministic smoke slice only; not independent evaluation",
            "elapsed_seconds": time.monotonic(),
        }
        (output / "training_metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return metadata
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--odapi-root", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    metadata = train_session_a_demo_detector(
        args.data_root, args.output, odapi_root=args.odapi_root,
        base_checkpoint=args.base_checkpoint, seed=args.seed,
        train_steps=args.train_steps, batch_size=args.batch_size,
    )
    print(json.dumps({key: metadata[key] for key in ("usage", "detector_path", "detector_sha256")}, indent=2))


if __name__ == "__main__":
    main()
