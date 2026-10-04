"""Float Keras to TFLite export with canonical index-ordered labels."""

import argparse
import json
from pathlib import Path
import tempfile

from .labels import LABELS


def write_labels(path):
    path = Path(path)
    path.write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    return path


def export_model(model_path, output_directory, training_metadata, *, smoke=False):
    import tensorflow as tf
    from .inspect_litert import inspect_tflite
    from .classifier_contract import build_contract
    model_path = Path(model_path)
    metadata = json.loads(Path(training_metadata).read_text(encoding="utf-8"))
    if metadata.get("preprocessing") != "MobileNetV3Small include_preprocessing=True; RGB float32 0..255":
        raise ValueError("Training preprocessing metadata missing or inconsistent")
    if smoke:
        if metadata.get("evaluation_report") != "SYNTHETIC-SMOKE-ONLY":
            raise ValueError("Synthetic smoke export must be explicitly labeled")
    elif (metadata.get("preflight_status") != "READY" or
          Path(metadata.get("selected_checkpoint", "")).resolve() != model_path.resolve() or
          not metadata.get("evaluation_report") or
          not Path(metadata["evaluation_report"]).is_file()):
        raise ValueError("Real export requires READY provenance, selected checkpoint and test evaluation")
    if not smoke and json.loads(Path(metadata["evaluation_report"]).read_text(encoding="utf-8")).get("evaluated_split") != "test":
        raise ValueError("Real export requires a final TEST evaluation report")
    model = tf.keras.models.load_model(model_path)
    if model.output_shape[-1] != 53:
        raise ValueError("Selected Keras model is not 53-way")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    tflite = output / "card_classifier.tflite"
    # TF 2.16/Keras 3 can abort inside from_keras_model on macOS; the
    # forward-only SavedModel path is the supported converter boundary.
    with tempfile.TemporaryDirectory() as directory:
        model.export(directory)
        converter = tf.lite.TFLiteConverter.from_saved_model(directory)
        converter.optimizations = []
        tflite.write_bytes(converter.convert())
    labels = write_labels(output / "labels.txt")
    inspect_tflite(tflite)  # Reject an unusable or wrong-width export.
    build_contract(tflite, metadata, labels, metadata.get("evaluation_report"), output / "classifier.json")
    return tflite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_model(args.model, args.output, args.training_metadata))


if __name__ == "__main__":
    main()
