"""Inspect actual detector TFLite bytes and run a zero-input diagnostic inference."""

import argparse
import hashlib
import json
from pathlib import Path


def _tensor(detail):
    import numpy as np
    parameters = detail["quantization_parameters"]
    return {
        "name": detail["name"], "index": int(detail["index"]),
        "shape": [int(value) for value in detail["shape"]],
        "shape_signature": [int(value) for value in detail["shape_signature"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "quantization": {
            "scale": float(detail["quantization"][0]),
            "zero_point": int(detail["quantization"][1]),
            "scales": [float(value) for value in parameters["scales"]],
            "zero_points": [int(value) for value in parameters["zero_points"]],
            "quantized_dimension": int(parameters["quantized_dimension"]),
        },
    }


def inspect_artifact(path):
    from ai_edge_litert.interpreter import Interpreter
    import numpy as np

    path = Path(path)
    content = path.read_bytes()
    if len(content) < 8 or content[4:8] != b"TFL3":
        raise ValueError("Expected a TFLite flatbuffer artifact")
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if not inputs or not outputs:
        raise ValueError("TFLite artifact has no input or output tensors")
    for detail in inputs:
        interpreter.set_tensor(detail["index"],
                               np.zeros(detail["shape"], dtype=detail["dtype"]))
    interpreter.invoke()
    observed = []
    raw_values = {}
    for detail in outputs:
        value = interpreter.get_tensor(detail["index"])
        raw_values[detail["name"]] = value
        observed.append({
            "name": detail["name"], "shape": list(value.shape),
            "minimum": float(value.min()) if value.size else None,
            "maximum": float(value.max()) if value.size else None,
        })
    signature_to_raw = {}
    signatures = interpreter.get_signature_list()
    if len(inputs) == 1:
        sample = np.zeros(inputs[0]["shape"], dtype=inputs[0]["dtype"])
        for signature_name, signature in signatures.items():
            if len(signature["inputs"]) != 1:
                continue
            signature_values = interpreter.get_signature_runner(signature_name)(
                **{signature["inputs"][0]: sample})
            signature_to_raw[signature_name] = {
                key: [name for name, raw in raw_values.items()
                      if raw.shape == value.shape and np.array_equal(raw, value)]
                for key, value in signature_values.items()}
    return {
        "artifact_sha256": hashlib.sha256(content).hexdigest(),
        "file_size_bytes": len(content),
        "input_tensors": [_tensor(detail) for detail in inputs],
        "output_tensors": [_tensor(detail) for detail in outputs],
        "signatures": signatures,
        "signature_to_raw": signature_to_raw,
        "observed_outputs": observed,
        "contains_detection_postprocess_custom_op":
            b"TFLite_Detection_PostProcess" in content,
        "output_semantics": "UNKNOWN",
        "nms_ownership": "UNKNOWN",
        "qualification": "Zero-input smoke values and tensor names alone do not establish output semantics.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = inspect_artifact(args.model)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
