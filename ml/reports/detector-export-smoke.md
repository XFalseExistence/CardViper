# One-class detector export smoke, 2026-10-05

**Result: PASS for tooling only.** A random-weight SSD MobileNet V2 FPNLite
320 with exactly one semantic class, `CARD`, instantiated, executed a synthetic
forward pass, saved/restored a checkpoint, exported a SavedModel, converted to
float TFLite, loaded in standalone LiteRT, and ran synthetic zero-input
inference. No real detector was trained, evaluated, selected, or published.
Synthetic output values are not CardViper accuracy metrics.

## Pinned environment

- macOS x86_64, Python 3.11.17
- TensorFlow 2.16.2; `tf-models-official` 2.16.0; `tf-keras` 2.16.0 with
  `TF_USE_LEGACY_KERAS=1`; `tensorflow-io` 0.37.1
- TensorFlow Models Object Detection API source:
  [`930a6f98f7debcc32ca7afbca4a176dbf9211e03`](https://github.com/tensorflow/models/commit/930a6f98f7debcc32ca7afbca4a176dbf9211e03),
  sparse paths `research/object_detection` and `research/slim`
- `grpcio-tools` 1.62.3 compiled the OD API protobufs;
  `ai-edge-litert` 1.0.1 loaded and invoked the export
- Full 127-package freeze: [detector-smoke-requirements.txt](detector-smoke-requirements.txt).
  It describes this macOS x86_64 smoke host, not Android runtime dependencies.

The official `v2.16.0` Models tag contains `official/` but not the
`research/object_detection` tree, so the OD API source is pinned separately.
The OD API's [mobile export guide](https://github.com/tensorflow/models/blob/930a6f98f7debcc32ca7afbca4a176dbf9211e03/research/object_detection/g3doc/running_on_mobile_tf2.md)
and [FPNLite 320 pipeline](https://github.com/tensorflow/models/blob/930a6f98f7debcc32ca7afbca4a176dbf9211e03/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config)
were the source of the smoke model. The script changes `model.ssd.num_classes`
from 90 to 1 in a private copy of the config; it does not download weights or
train on Joshua data.

To reproduce, create an ignored/external Python 3.11 venv, install the pinned
requirements, clone the cited Models commit, and compile its protos:

```sh
python3.11 -m venv .venv-detector
.venv-detector/bin/python -m pip install -r ml/reports/detector-smoke-requirements.txt
git clone --filter=blob:none --sparse https://github.com/tensorflow/models.git /path/to/models
git -C /path/to/models checkout 930a6f98f7debcc32ca7afbca4a176dbf9211e03
git -C /path/to/models sparse-checkout set research/object_detection research/slim
.venv-detector/bin/python -m grpc_tools.protoc -I/path/to/models/research \
  --python_out=/path/to/models/research /path/to/models/research/object_detection/protos/*.proto
TF_USE_LEGACY_KERAS=1 PYTHONPATH=ml .venv-detector/bin/python \
  -m cardviper_ml.detector_export_smoke --odapi-root /path/to/models \
  --output ml/local/detector-smoke
PYTHONPATH=ml .venv-detector/bin/python -m cardviper_ml.inspect_detector_litert \
  --model ml/local/detector-smoke/detector-smoke.tflite \
  --report ml/local/detector-smoke/inspection.json
```

The output path must not exist before the smoke run. The entire smoke output
belongs under ignored `ml/local/`; no random-weight model is a production
asset. The isolated environment needed a prebuilt `cryptography==45.0.7`
wheel on this host. An earlier install attempt failed while trying to build
`cryptography==50.0.2` into a denied user cache; that was a setup error, not
a model/export failure.

## Actual inspected smoke artifact

The first successful isolated probe's `detector-smoke.tflite` was 10,342,540
bytes, SHA-256
`c2fb72a7a5888f1f0e20ff00d727b4565a102b4a7c4e98658bcf5f588f914678`.
Random-weight rebuilds may have different bytes. The reusable CLI also passed
its full integration test, including standalone LiteRT inference.

| Role | Actual tensor name | Shape | Dtype | Zero-input range |
| --- | --- | --- | --- | --- |
| Input | `serving_default_input:0` | `[1,320,320,3]` | float32 | zero fixture |
| Score-like | `StatefulPartitionedCall:1` | `[1,10]` | float32 | 0.0099518 |
| Boxes | `StatefulPartitionedCall:3` | `[1,10,4]` | float32 | -0.0875 to 0.13321 |
| Count | `StatefulPartitionedCall:0` | `[1]` | float32 | 10 |
| Class-like | `StatefulPartitionedCall:2` | `[1,10]` | float32 | 0 |

All tensor quantization scales and zero points were zero. The interpreter
signature maps `output_0`, `output_1`, `output_2`, `output_3` to raw
`StatefulPartitionedCall:0`, `:1`, `:2`, `:3`, respectively, by equality
of actual zero-input outputs. The OD API
[exporter source](https://github.com/tensorflow/models/blob/930a6f98f7debcc32ca7afbca4a176dbf9211e03/research/object_detection/export_tflite_graph_lib_tf2.py)
returns count, scores, classes, then boxes to the TFLite signature. Together
these establish the smoke output roles. The actual FlatBuffer contains
`TFLite_Detection_PostProcess`, and standalone LiteRT invoked it, so NMS is
embedded in this **smoke** artifact.

The zero-input sample cannot establish box coordinate ordering, normalized
versus model-pixel coordinates, useful thresholds, or real-image behavior.
Those remain **UNKNOWN**. The production contract is deliberately only
[a template](../contracts/detector.template.json); `ml/contracts/detector.json`
does not exist. Its values must come from the later selected real export and
held-out evaluation, not this smoke model.
