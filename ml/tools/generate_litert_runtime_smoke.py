#!/usr/bin/env python3
"""Rebuild the deterministic TEST_ONLY arithmetic model used by Android runtime smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from ai_edge_litert.interpreter import Interpreter
import tensorflow as tf
import numpy as np


PINNED_TENSORFLOW = "2.16.2"
PINNED_LITERT_INSPECTOR = "1.0.1"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "app/src/androidTest/assets/vision/litert_runtime_smoke.TEST_ONLY.tflite"
)


def build_model() -> bytes:
    if tf.__version__ != PINNED_TENSORFLOW:
        raise RuntimeError(
            f"Generate this fixture with tensorflow=={PINNED_TENSORFLOW}; found {tf.__version__}"
        )

    class ScaleByTwo(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 2, 2, 3], tf.float32, name="image")])
        def serve(self, image: tf.Tensor) -> dict[str, tf.Tensor]:
            return {"scaled": image * tf.constant(2.0, dtype=tf.float32)}

    module = ScaleByTwo()
    concrete = module.serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def inspect(model_bytes: bytes) -> dict[str, object]:
    inspector_version = importlib.metadata.version("ai-edge-litert")
    if inspector_version != PINNED_LITERT_INSPECTOR:
        raise RuntimeError(
            f"Inspect this fixture with ai-edge-litert=={PINNED_LITERT_INSPECTOR}; found {inspector_version}"
        )
    interpreter = Interpreter(model_content=model_bytes, num_threads=1)
    interpreter.allocate_tensors()

    def tensor(details: dict[str, object]) -> dict[str, object]:
        dtype = details["dtype"]
        if np.dtype(dtype) != np.dtype(np.float32):
            raise RuntimeError(f"Expected float32 synthetic tensor, found {dtype}")
        return {
            "name": details["name"],
            "shape": [int(dimension) for dimension in details["shape"]],
            "dtype": "FLOAT32",
        }

    return {
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "fixture_kind": "TEST_ONLY_ARITHMETIC_SCALE_BY_TWO_NO_CARD_SEMANTICS",
        "generator": f"tensorflow=={PINNED_TENSORFLOW}",
        "inspector": f"ai-edge-litert=={PINNED_LITERT_INSPECTOR}",
        "signature": {
            "input": tensor(interpreter.get_input_details()[0]),
            "outputs": [tensor(detail) for detail in interpreter.get_output_details()],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model_bytes = build_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model_bytes)
    metadata = inspect(model_bytes)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"model={args.output}")
    print(f"metadata={metadata_path}")
    print(f"sha256={metadata['model_sha256']}")
    print(f"bytes={len(model_bytes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
