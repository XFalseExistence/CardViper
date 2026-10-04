"""Guarded two-stage MobileNetV3Small training; real data only after READY."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from .classifier_config import ClassifierDataPaths, ClassifierTrainingConfig
from .classifier_dataset import load_crop_split, rgb_array, seeded_batches
from .classifier_augmentation import augment_image
from .classifier_model import ModelConfig, build_model, compile_stage, model_spec
from .classifier_preflight import audit_classifier_data
from .labels import LABELS
from .splits import read_splits


def require_ready(report):
    if not isinstance(report, dict):
        report = json.loads(Path(report).read_text(encoding="utf-8"))
    if report.get("status") != "READY" or report.get("ready") is not True:
        raise ValueError("Real training requires cardviper-classifier-preflight == READY")
    return report


def _crop_key(row):
    return row.source_id, row.source_image_sha256, row.annotation_id, row.label


def validate_crop_coverage(expected, actual):
    if expected != actual:
        raise ValueError(f"Crop coverage differs from approved split annotations: missing={len(expected-actual)}, extra={len(actual-expected)}")


def _expected_crops(rows):
    return {(row.source_id, row.image_sha256, obj.annotation_id, obj.label)
            for row in rows for obj in row.objects}


def _dataset(samples, *, split, config, batch_size):
    import numpy as np
    import tensorflow as tf
    epoch = 0

    def generate():
        nonlocal epoch
        current = epoch
        epoch += 1
        order = [sample for batch in seeded_batches(samples, batch_size, config.seed + current) for sample in batch] if split == "train" else samples
        for index, sample in enumerate(order):
            image = rgb_array(sample, config.input_size)
            if split == "train":
                image = augment_image(image, split="train", seed=config.seed + current * 1000003 + index)
            yield np.asarray(image, dtype=np.float32), np.int32(sample.label_index)

    dataset = tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((config.input_size, config.input_size, 3), tf.float32),
        tf.TensorSpec((), tf.int32)))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_classifier(sources_path, splits_directory, crop_root, output_directory, *,
                     config=ModelConfig(), batch_size=32, head_epochs=10, fine_tune_epochs=10):
    """Return metadata; test/holdout crops are never opened during fitting."""
    if batch_size <= 0 or head_epochs <= 0 or fine_tune_epochs <= 0:
        raise ValueError("Positive batch size and epoch counts required")
    preflight = require_ready(audit_classifier_data(sources_path, splits_directory))
    splits = read_splits(splits_directory)
    paths = ClassifierDataPaths.from_split_directory(splits_directory)
    ClassifierTrainingConfig(labels=LABELS, declared_output_width=53, seed=config.seed, data=paths)
    crop_root = Path(crop_root)
    train = load_crop_split(crop_root / "train", "train")
    val = load_crop_split(crop_root / "val", "val")
    for split, samples in (("train", train), ("val", val)):
        actual_list = [_crop_key(row) for row in samples]
        if len(actual_list) != len(set(actual_list)):
            raise ValueError(f"Duplicate {split} crop provenance")
        validate_crop_coverage(_expected_crops(splits[split]), set(actual_list))
    if not train or not val:
        raise ValueError("Train and validation crops must be nonempty")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config.seed)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model, backbone = build_model(config)
    train_ds = _dataset(train, split="train", config=config, batch_size=batch_size)
    val_ds = _dataset(val, split="val", config=config, batch_size=batch_size)

    def callbacks(path):
        return [tf.keras.callbacks.ModelCheckpoint(str(path), monitor="val_loss", mode="min", save_best_only=True),
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3, restore_best_weights=True)]

    stage1 = output / "stage1_best.keras"
    stage2 = output / "stage2_best.keras"
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                   steps_per_epoch=math.ceil(len(train)/batch_size), validation_steps=math.ceil(len(val)/batch_size),
                   callbacks=callbacks(stage1), verbose=2)
    model = tf.keras.models.load_model(stage1)
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    compile_stage(model, backbone, config, fine_tune=True)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs,
                   steps_per_epoch=math.ceil(len(train)/batch_size), validation_steps=math.ceil(len(val)/batch_size),
                   callbacks=callbacks(stage2), verbose=2)
    chosen = stage1 if min(h1.history["val_loss"]) <= min(h2.history["val_loss"]) else stage2
    labels_bytes = ("\n".join(LABELS) + "\n").encode("utf-8")
    metadata = {**model_spec(config), "schema_version": 1,
                "selected_checkpoint": str(chosen), "selected_by": "minimum validation loss",
                "elapsed_seconds": time.monotonic() - started,
                "batch_size": batch_size, "head_epochs_limit": head_epochs,
                "fine_tune_epochs_limit": fine_tune_epochs,
                "stage1_history": h1.history, "stage2_history": h2.history,
                "labels_sha256": hashlib.sha256(labels_bytes).hexdigest(),
                "split_manifest_sha256": {name: hashlib.sha256((Path(splits_directory) / f"{name}.jsonl").read_bytes()).hexdigest()
                                          for name in ("train", "val", "test", "holdout")},
                "crop_manifest_sha256": {name: hashlib.sha256((crop_root / name / "crops.jsonl").read_bytes()).hexdigest()
                                         for name in ("train", "val")},
                "source_ids": sorted(preflight["permitted_source_ids"]),
                "preflight_status": preflight["status"],
                "model_config": asdict(config)}
    (output / "training_metadata.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    metadata = train_classifier(args.sources, args.splits, args.crops, args.output,
                                config=ModelConfig(input_size=args.input_size, seed=args.seed), batch_size=args.batch_size)
    print(f"Selected checkpoint: {metadata['selected_checkpoint']}")


if __name__ == "__main__":
    main()
