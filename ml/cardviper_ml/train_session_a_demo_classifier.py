"""Train a 52-face, Session-A-only MobileNetV3Small demonstration classifier."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from .classifier_dataset import rgb_array, verify_crop_provenance
from .labels import LABELS
from .manifest import read_manifest


DEMO_LABELS = LABELS[:-1]
DEMO_USAGE = "SESSION_A_DEMO_ONLY"
INDEPENDENCE = "NOT_INDEPENDENT"
PREPROCESSING = "MobileNetV3Small include_preprocessing=True; RGB float32 0..255"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_face_only_labels(labels) -> tuple[str, ...]:
    """Return labels only when every entry is a known face-up identity."""
    labels = tuple(labels)
    if "BACK" in labels:
        raise ValueError("Session A demo supports face-up card labels only")
    if any(label not in DEMO_LABELS for label in labels):
        raise ValueError("Session A demo labels must be canonical face identities")
    return labels


def build_demo_metadata(*, data_root: Path, seed: int, epochs: int, batch_size: int,
                        train_sample_count: int, smoke_sample_count: int) -> dict:
    """Create non-release evidence metadata without reading a model or evaluation."""
    if type(seed) is not int or any(type(value) is not int or value <= 0
                                   for value in (epochs, batch_size, train_sample_count, smoke_sample_count)):
        raise ValueError("seed and positive training counts are required")
    return {
        "schema_version": 1,
        "usage": DEMO_USAGE,
        "independence": INDEPENDENCE,
        "independent_evaluation": False,
        "face_up_only": True,
        "back_supported": False,
        "supported_labels": list(DEMO_LABELS),
        "output_width": len(DEMO_LABELS),
        "architecture": "MobileNetV3Small",
        "input_layout": "NHWC",
        "preprocessing": PREPROCESSING,
        "data_root": str(Path(data_root)),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_sample_count": train_sample_count,
        "smoke_sample_count": smoke_sample_count,
        "smoke_note": "Session A deterministic smoke slice only; not independent evaluation",
    }


def _read_demo_input(data_root: Path) -> tuple[dict, Path, list, list]:
    data_root = Path(data_root)
    try:
        source = json.loads((data_root / "demo_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Missing or invalid Session A demo metadata") from error
    if (source.get("usage") != DEMO_USAGE or source.get("independence") != INDEPENDENCE or
            source.get("independent_evaluation") is not False or source.get("face_up_only") is not True or
            source.get("back_supported") is not False or source.get("classifier_labels") != list(DEMO_LABELS)):
        raise ValueError("Session A demo metadata does not describe the approved face-only input")
    source_root = Path(source.get("source_root", ""))
    if not source_root.is_dir():
        raise ValueError("Session A source root is unavailable")
    train = read_manifest(data_root / "train.jsonl")
    smoke = read_manifest(data_root / "smoke.jsonl")
    if not train or not smoke:
        raise ValueError("Session A demo train and smoke manifests must be nonempty")
    validate_face_only_labels(obj.label for row in (*train, *smoke) for obj in row.objects)
    return source, source_root, train, smoke


def _dataset(samples, *, input_size: int, batch_size: int, seed: int, training: bool):
    import numpy as np
    import tensorflow as tf

    ordered = list(samples)
    if training:
        rng = np.random.default_rng(seed)
        rng.shuffle(ordered)

    def generate():
        for sample in ordered:
            yield (np.asarray(rgb_array(sample, input_size), dtype=np.float32),
                   np.int32(DEMO_LABELS.index(sample.label)))

    return tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((input_size, input_size, 3), tf.float32),
        tf.TensorSpec((), tf.int32),
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_model(*, input_size: int, seed: int):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(input_size, input_size, 3), name="rgb_0_255")
    backbone = tf.keras.applications.MobileNetV3Small(
        include_top=False, weights="imagenet", include_preprocessing=True,
        input_shape=(input_size, input_size, 3),
    )
    backbone.trainable = False
    features = backbone(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(len(DEMO_LABELS), activation="softmax", name="session_a_face_identity")(pooled)
    model = tf.keras.Model(inputs, outputs, name="cardviper_session_a_face_demo")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="top1")])
    return model


def train_session_a_demo_classifier(data_root: Path, output: Path, *, seed: int, epochs: int,
                                    batch_size: int, input_size: int = 224) -> dict:
    """Fit only Session A face crops and write explicit non-promotable metadata."""
    if input_size < 96:
        raise ValueError("input_size must be at least 96")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    source, source_root, train_rows, smoke_rows = _read_demo_input(Path(data_root))
    roots = {"playing-cards-seed": source_root}
    train = verify_crop_provenance(Path(data_root) / "classifier_crops" / "train", train_rows, roots, "train")
    smoke = verify_crop_provenance(Path(data_root) / "classifier_crops" / "smoke", smoke_rows, roots, "holdout")
    validate_face_only_labels(sample.label for sample in (*train, *smoke))
    metadata = build_demo_metadata(
        data_root=Path(data_root), seed=seed, epochs=epochs, batch_size=batch_size,
        train_sample_count=len(train), smoke_sample_count=len(smoke),
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        import tensorflow as tf

        started = time.monotonic()
        model = _build_model(input_size=input_size, seed=seed)
        train_ds = _dataset(train, input_size=input_size, batch_size=batch_size, seed=seed, training=True)
        smoke_ds = _dataset(smoke, input_size=input_size, batch_size=batch_size, seed=seed, training=False)
        checkpoint = output / "session_a_face_demo.keras"
        history = model.fit(
            train_ds,
            validation_data=smoke_ds,
            epochs=epochs,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", mode="min", save_best_only=True)],
            verbose=2,
        )
        metadata.update({
            "input_size": input_size,
            "selected_checkpoint": str(checkpoint),
            "selected_checkpoint_sha256": _sha256(checkpoint),
            "selected_by": "minimum Session A smoke loss; not independent evaluation",
            "elapsed_seconds": time.monotonic() - started,
            "history": history.history,
            "archive_sha256": source["archive_sha256"],
            "input_metadata_sha256": _sha256(Path(data_root) / "demo_metadata.json"),
            "train_manifest_sha256": _sha256(Path(data_root) / "train.jsonl"),
            "smoke_manifest_sha256": _sha256(Path(data_root) / "smoke.jsonl"),
            "train_crops_sha256": _sha256(Path(data_root) / "classifier_crops" / "train" / "crops.jsonl"),
            "smoke_crops_sha256": _sha256(Path(data_root) / "classifier_crops" / "smoke" / "crops.jsonl"),
        })
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()
    metadata = train_session_a_demo_classifier(
        args.data_root, args.output, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, input_size=args.input_size,
    )
    print(json.dumps({key: metadata[key] for key in ("usage", "output_width", "selected_checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
