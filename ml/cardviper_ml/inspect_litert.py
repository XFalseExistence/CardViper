"""Inspect actual TFLite bytes, with no guessed tensor values."""

import argparse
import hashlib
import json
from pathlib import Path


def validate_tensor_report(report):
    if len(report["input_tensors"]) != 1 or len(report["output_tensors"]) != 1:
        raise ValueError("Classifier requires exactly one input and one output tensor")
    if report["output_tensors"][0]["shape"][-1] != 53:
        raise ValueError("Classifier output width must be 53")
    if len(report["input_tensors"][0]["shape"]) != 4:
        raise ValueError("Classifier input must be rank 4")
    return report


def _tensor(detail):
    quant = detail.get("quantization_parameters", {})
    return {"index": int(detail["index"]), "name": detail["name"],
            "shape": [int(x) for x in detail["shape"]], "dtype": detail["dtype"].__name__,
            "quantization": {"scales": [float(x) for x in quant.get("scales", [])],
                             "zero_points": [int(x) for x in quant.get("zero_points", [])],
                             "quantized_dimension": int(quant.get("quantized_dimension", 0))}}


def inspect_tflite(path):
    path = Path(path)
    data = path.read_bytes()
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError as error:
            raise RuntimeError("Install the train extra or ai-edge-litert to inspect a real model") from error
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    report = {"schema_version": 1, "model_sha256": hashlib.sha256(data).hexdigest(),
              "model_bytes": len(data),
              "input_tensor_count": len(interpreter.get_input_details()),
              "output_tensor_count": len(interpreter.get_output_details()),
              "input_tensors": [_tensor(row) for row in interpreter.get_input_details()],
              "output_tensors": [_tensor(row) for row in interpreter.get_output_details()]}
    return validate_tensor_report(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(inspect_tflite(args.model), sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
