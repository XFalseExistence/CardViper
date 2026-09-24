# CardViper V0.2 Phase B Models + Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the completed V0.2 Phase A image-test pipeline into real, fully offline multi-card recognition on the Pixel 7 by training/exporting a one-class card detector and a 53-way face/BACK classifier, then integrating both through the existing Android vision interfaces.

**Architecture:** Keep the approved two-model boundary: a wide detector returns source-image `CARD` boxes; the engine crops from the bounded decoded source image; a 53-way classifier returns one of 52 exact identities or `BACK`. Training/export lives under `ml/`; Android inference remains behind `CardDetector` and `CardRecognizer`. Freeze tensor/preprocessing/postprocessing contracts only from actual exported artifacts, then use standalone packaged LiteRT on Android with CPU correctness as the baseline before optional acceleration.

**Tech Stack:** Python 3.11; TensorFlow/Keras for classifier training; TensorFlow Object Detection API SSD MobileNet V2 FPNLite as the preferred detector baseline after an export smoke gate; LiteRT-compatible `.tflite` exports; `ai-edge-litert` for artifact inspection; Android Kotlin/Java 17; standalone LiteRT Android runtime; JUnit + pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-cardviper-v0-2-image-recognition-design.md`

**Contract:** `docs/vision/cardviper-v0-2-model-contract.md`

## Global Constraints

- Primary device: Google Pixel 7.
- Android: compileSdk 37, targetSdk 36, minSdk 26, Java 17.
- Fully offline at app runtime: no cloud inference, model download, telemetry, auth, ads, or network dependency.
- Preserve existing `CardDetector` and `CardRecognizer` interfaces.
- Detector has exactly one semantic object class: `CARD`.
- Classifier has exactly 53 outputs: 52 exact rank/suit identities plus `BACK`.
- `CardLabelCodec` is the authoritative label source; output index order must be explicit and tested.
- Do not guess tensor dimensions, dtype, normalization, quantization, output layout, NMS ownership, or output ordering before inspecting exported artifacts.
- Image-test recognition remains diagnostic only: no haptics, no CameraX recognition, no tracking, and no ledger/session mutation.
- Low-confidence face predictions remain visible and count in the still-image diagnostic preview; `BACK` contributes zero.
- Training/validation/test splits must be grouped by source/scene/session, never random near-duplicate frame splits.
- Pixel 7 hard-case acceptance images remain held out from training until acceptance testing.
- Do not commit third-party dataset images or private/user photos to git. Commit only tooling, manifests, licenses/attribution records, small synthetic test fixtures, reports, and deliberately selected model artifacts.
- Do not use Ultralytics code, training pipelines, or trained/fine-tuned Ultralytics models unless licensing is separately approved.
- Stable Android debug signing must remain unchanged.

## Review Focus

1. **Leakage between train and evaluation:** split validator must reject the same source/session appearing in more than one split.
2. **Label-order mismatch:** export tests must prove all 53 labels and exact classifier output-index mapping, not just set equality.
3. **Detector resize/letterbox reversal:** Android tests must prove model-space boxes map back to source pixels correctly at edges and non-square source sizes.
4. **Runtime/model contract drift:** Android model loading must fail clearly when packaged asset shape/dtype/output count differs from the frozen contract.
5. **False operational success:** Pixel acceptance must score whole-image correctness and false positives, not merely per-card top-1 accuracy.

---

### Task B1: ML/Data Foundation

**Throttle:** HIGH

**Files:**
- Create: `ml/README.md`
- Create: `ml/pyproject.toml`
- Create: `ml/cardviper_ml/__init__.py`
- Create: `ml/cardviper_ml/labels.py`
- Create: `ml/cardviper_ml/manifest.py`
- Create: `ml/cardviper_ml/import_roboflow.py`
- Create: `ml/cardviper_ml/splits.py`
- Create: `ml/cardviper_ml/build_classifier_crops.py`
- Create: `ml/datasets/sources.json`
- Create: `ml/tests/test_labels.py`
- Create: `ml/tests/test_manifest.py`
- Create: `ml/tests/test_splits.py`
- Create: `ml/tests/test_classifier_crops.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: CardViper canonical labels from `CardLabelCodec` conventions (`A..K` + `C/D/H/S`, plus `BACK`).
- Produces: canonical `labels.txt`; normalized dataset manifest rows with `source_id`, `scene_id`, `image_path`, `label`, and normalized/source box data; deterministic grouped train/val/test split files; classifier crop directories/manifests.

- [ ] **Step 1: Write label tests before implementation.** Assert exactly 53 unique labels, exactly one `BACK`, every rank/suit pair, stable explicit order, and codec-compatible strings such as `AC`, `10H`, `KS`.
- [ ] **Step 2: Run `python -m pytest ml/tests/test_labels.py -q` and verify failure.**
- [ ] **Step 3: Implement `labels.py` with an explicit canonical ordered tuple and a CLI that writes `labels.txt`.** Do not derive output order from Python/Kotlin enum iteration.
- [ ] **Step 4: Add manifest/import tests.** Use tiny generated fixtures; verify Roboflow/YOLO face labels map to exact CardViper labels, detector view collapses all face identities to `CARD`, invalid/out-of-range boxes fail, and BACK remains a classifier identity rather than a fake face.
- [ ] **Step 5: Implement manifest/import code.** Input datasets remain outside git; `sources.json` records source name, URL, license, attribution text, local root alias, and whether it may be used for detector/classifier training.
- [ ] **Step 6: Add split leakage tests.** Assert the same `source_id + scene_id` cannot occur across train/val/test and that deterministic seed reruns produce identical split files.
- [ ] **Step 7: Implement grouped split generation and validator.** Fail closed on missing grouping metadata.
- [ ] **Step 8: Add crop tests and implement classifier crop generation.** Clamp boxes safely, preserve exact identity, reject empty crops, and write crop provenance back to the manifest.
- [ ] **Step 9: Add `.gitignore` rules for raw datasets, generated crops, training runs, checkpoints, exported models, and local virtual environments.** Keep source manifests and tests tracked.
- [ ] **Step 10: Document the seed-data workflow.** Record the currently selected CC BY 4.0 52-class Playing Cards dataset as a seed source, state that it lacks `BACK`, and require CardViper-owned or separately licensed BACK examples plus no-card detector negatives.
- [ ] **Step 11: Run `python -m pytest ml/tests -q`, then existing Android verification (`./gradlew test`, `assembleDebugAndroidTest`, `lintDebug`, `assembleDebug`).**
- [ ] **Step 12: Commit and push:** `feat: add CardViper ML data foundation`. STOP for review.

### Task B2: 53-Way Classifier Baseline + Real Export Contract

**Throttle:** HIGH

**Files:**
- Create: `ml/cardviper_ml/train_classifier.py`
- Create: `ml/cardviper_ml/eval_classifier.py`
- Create: `ml/cardviper_ml/export_classifier.py`
- Create: `ml/cardviper_ml/inspect_litert.py`
- Create: `ml/tests/test_classifier_export.py`
- Create: `ml/contracts/classifier.json`
- Create: `ml/reports/classifier-baseline.md`
- Create after successful training/export: `ml/exports/classifier/labels.txt`
- Create after successful training/export: `ml/exports/classifier/card_classifier.tflite`

**Interfaces:**
- Consumes: B1 grouped classifier manifests/crops and canonical labels.
- Produces: a real 53-output LiteRT-compatible classifier artifact, ordered label file, measured metrics, and machine-readable frozen tensor/preprocessing/output contract.

- [ ] **Step 1: Add a training-config test** proving the classifier output width is 53 and dataset split groups are respected.
- [ ] **Step 2: Implement a MobileNetV3Small transfer-learning baseline** with a 53-way softmax head. Begin with a float model and deterministic seed; augment rotation, scale, mild perspective, exposure/contrast, blur/compression, crop-padding variation, and partial edge/corner occlusion without changing identity.
- [ ] **Step 3: Train using the B1 train split only.** Validation selects checkpoints; test split remains untouched until final baseline evaluation.
- [ ] **Step 4: Evaluate exact top-1, per-class recall, BACK accuracy, confusion matrix, and uncertainty/coverage at candidate thresholds.** Save metrics in `classifier-baseline.md`.
- [ ] **Step 5: Export the best float model to `.tflite`.** Do not hand-write Android tensor assumptions.
- [ ] **Step 6: Inspect the actual exported artifact with LiteRT tooling.** Record exact input shape/layout/dtype, normalization expected by the trained model, output shape/dtype, and output index-to-label mapping in `classifier.json`.
- [ ] **Step 7: Add export tests.** Load the exported artifact, assert 53 outputs, assert every `labels.txt` line round-trips to a valid CardViper identity, and assert each output index maps to its recorded label.
- [ ] **Step 8: Run classifier tests + full Android verification.**
- [ ] **Step 9: Commit and push:** `feat: train CardViper 53-way classifier baseline`. STOP for review.

### Task B3: One-Class Detector Export Gate + Baseline

**Throttle:** HIGH

**Files:**
- Create: `ml/cardviper_ml/build_detector_records.py`
- Create: `ml/cardviper_ml/detector_export_smoke.py`
- Create: `ml/cardviper_ml/train_detector.py`
- Create: `ml/cardviper_ml/eval_detector.py`
- Create: `ml/cardviper_ml/export_detector.py`
- Create: `ml/tests/test_detector_dataset.py`
- Create: `ml/tests/test_detector_export.py`
- Create: `ml/contracts/detector.json`
- Create: `ml/reports/detector-baseline.md`
- Create after successful training/export: `ml/exports/detector/card_detector.tflite`

**Interfaces:**
- Consumes: B1 full-scene manifests with all face labels collapsed to one `CARD` class plus no-card negatives.
- Produces: one-class detector artifact and a frozen real output contract including resize/letterbox and NMS semantics.

- [ ] **Step 1: Test detector dataset conversion.** Assert exactly one semantic class, valid boxes, retained negative scenes, deterministic grouped splits, and no classifier identity leakage into detector class IDs.
- [ ] **Step 2: Before full training, run an export smoke gate** using the preferred TensorFlow Object Detection API SSD MobileNet V2 FPNLite 320 baseline. Prove that a one-class model can be instantiated, exported to `.tflite`, loaded by LiteRT tooling, and its outputs inspected.
- [ ] **Step 3: If the smoke gate fails because the current supported TensorFlow/OD-API combination cannot produce a runnable LiteRT artifact, STOP and report the exact incompatibility.** Do not burn quota/compute on training and do not silently switch frameworks.
- [ ] **Step 4: Train the 320x320 one-class baseline** only after the export gate passes. Include full-table scenes, rotations, partial occlusions, chips/hands/table distractors, glare/shadows, scale variation, and no-card negatives.
- [ ] **Step 5: Evaluate detection recall, false positives per image, and whole-image detection correctness on the held-out detector test split.** mAP may be recorded but is not the acceptance metric by itself.
- [ ] **Step 6: Export the selected float detector and inspect the real contract.** Record input size/layout/dtype/normalization, box format, score/class/count tensors, model-space coordinate convention, NMS ownership, and score/NMS thresholds in `detector.json`.
- [ ] **Step 7: Add export/parser fixtures.** Verify output tensor counts/shapes and known synthetic box reversal through the recorded resize/letterbox transform.
- [ ] **Step 8: Escalate detector input from 320 to 640 only if held-out small-card/chest-level recall demonstrates that 320 is the bottleneck.** Record the evidence if changed.
- [ ] **Step 9: Run ML tests + full Android verification.**
- [ ] **Step 10: Commit and push:** `feat: train CardViper card detector baseline`. STOP for review.

### Task B4: Android LiteRT Runtime Adapters + Packaged Assets

**Throttle:** EXTRA HIGH

**Files:**
- Modify: `gradle/libs.versions.toml`
- Modify: `app/build.gradle.kts`
- Modify: `app/src/main/java/com/cardviper/app/CardViperApplication.kt`
- Create: `app/src/main/assets/vision/card_detector.tflite`
- Create: `app/src/main/assets/vision/card_classifier.tflite`
- Create: `app/src/main/assets/vision/labels.txt`
- Create: `app/src/main/assets/vision/detector_contract.json`
- Create: `app/src/main/assets/vision/classifier_contract.json`
- Create: `app/src/main/java/com/cardviper/app/vision/litert/LiteRtModelContract.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/litert/LiteRtImagePreprocessor.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/litert/LiteRtCardDetector.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/litert/LiteRtCardRecognizer.kt`
- Test: `app/src/test/java/com/cardviper/app/vision/litert/LiteRtModelContractTest.kt`
- Test: `app/src/test/java/com/cardviper/app/vision/litert/LiteRtImagePreprocessorTest.kt`
- Test: `app/src/test/java/com/cardviper/app/vision/litert/LiteRtDetectorPostprocessorTest.kt`
- Test: `app/src/test/java/com/cardviper/app/vision/litert/LiteRtClassifierPostprocessorTest.kt`
- Android test: `app/src/androidTest/java/com/cardviper/app/vision/litert/LiteRtAssetSmokeTest.kt`

**Interfaces:**
- Consumes: B2/B3 `.tflite` artifacts, label file, and frozen JSON contracts.
- Produces: concrete offline implementations of existing `CardDetector.detect(VisionImage)` and `CardRecognizer.recognize(VisionImage)`.

- [ ] **Step 1: Add standalone LiteRT Android dependency using the current stable standalone runtime.** Do not use Play Services model delivery/runtime because runtime must be self-contained offline.
- [ ] **Step 2: Write contract-loader failure tests first.** Wrong input shape, dtype, output count, missing label, duplicate label, wrong 53-label order, or incompatible contract must fail model initialization with a clear model-unavailable/incompatible error.
- [ ] **Step 3: Implement contract parsing and validation against packaged assets.** The JSON is generated from real B2/B3 exports; Kotlin must not invent missing values.
- [ ] **Step 4: Write image-preprocessing tests** using tiny RGB fixtures. Assert resize/letterbox, channel ordering, normalization, and dtype packing exactly match frozen contracts.
- [ ] **Step 5: Implement classifier runtime adapter** using LiteRT CompiledModel with CPU as the correctness baseline. Map the 53 output scores through packaged `labels.txt`; produce best identity, confidence, and top alternatives.
- [ ] **Step 6: Write detector postprocessing tests** for source-box reconstruction on landscape, portrait, edge-clamped, and non-square images.
- [ ] **Step 7: Implement detector runtime adapter** using the exported model's actual output contract and NMS ownership. Return source-image pixel `CardCandidate` objects.
- [ ] **Step 8: Add an instrumented packaged-asset smoke test** that loads both models on Android and runs deterministic fixture inference without network access.
- [ ] **Step 9: Replace production `UnavailableCardDetector`/`NoOpCardRecognizer` wiring only after the smoke test passes.** Keep fake seams for host tests.
- [ ] **Step 10: Run `./gradlew test`, `assembleDebugAndroidTest`, `lintDebug`, `assembleDebug`; verify no new network/storage permission; verify signer unchanged.**
- [ ] **Step 11: Commit and push:** `feat: enable offline CardViper LiteRT inference`. STOP for review and CI.

### Task B5: Pixel 7 Physical Acceptance + Threshold Calibration

**Throttle:** HIGH

**Files:**
- Create: `ml/eval/pixel7/manifest.json`
- Create: `ml/cardviper_ml/score_pixel_acceptance.py`
- Create: `ml/reports/pixel7-v0-2-acceptance.md`
- Modify only if calibration evidence requires it: application threshold configuration in the existing image-recognition engine/wiring.

**Interfaces:**
- Consumes: real APK from B4 and the held-out Pixel 7 hard-case set.
- Produces: reproducible acceptance report, calibrated thresholds, latency measurements, and an explicit pass/fail against V0.2 still-image success conditions.

- [ ] **Step 1: Build the held-out manifest** for the eight required cases: clean multi-card, chest-level FPV, rotated, partial occlusion, glare/shadow, repeated rank, face-down `BACK`, and no-card distractor. Record exact expected boxes/identities outside training data.
- [ ] **Step 2: Run every image through IMAGE TEST on Pixel 7.** Record detector candidates, exact identities, confidence, uncertainty, KO delta, KISS III delta, and end-to-end latency.
- [ ] **Step 3: Score operational metrics:** detection recall, false positives/image, exact face identity accuracy, BACK accuracy, uncertainty rate, and whole-image correctness (all intended cards correct with no extra card).
- [ ] **Step 4: Tune detector/classifier confidence thresholds only from calibration data, never from the final acceptance images.** Re-run acceptance once after thresholds are frozen.
- [ ] **Step 5: Verify repeated `CHOOSE ANOTHER IMAGE` runs remain stable and active shoe state is unchanged.**
- [ ] **Step 6: Record failures by category rather than hiding them.** If strong-perspective errors dominate, recommend the already-approved four-corner/projective-rectification refinement as the next milestone rather than weakening identity correctness.
- [ ] **Step 7: Run full Android verification and commit:** `test: record CardViper Pixel 7 V0.2 acceptance`. STOP for review.

### Task B6: Accuracy-Safe Optimization Gate

**Throttle:** HIGH; raise to EXTRA HIGH only for conversion/device-runtime failures.

**Files:**
- Create: `ml/cardviper_ml/quantize.py`
- Create: `ml/tests/test_quantized_parity.py`
- Create: `ml/reports/optimization.md`
- Modify packaged model assets only if a candidate is accepted.

**Interfaces:**
- Consumes: the accepted float B2/B3 artifacts and B5 metrics.
- Produces: either a measured decision to keep float models or a parity-verified smaller/faster packaged artifact.

- [ ] **Step 1: Treat float models as the reference.** Capture their Pixel 7 latency, model size, whole-image correctness, and per-card metrics.
- [ ] **Step 2: Try float16 first.** Use representative held-out-but-not-final-acceptance data; compare outputs and full evaluation metrics against float reference.
- [ ] **Step 3: Try full-int8 only if float16 does not meet the practical size/latency target.** Use representative calibration data and preserve exact output-index mapping.
- [ ] **Step 4: Reject any candidate that materially damages whole-image correctness or BACK/exact-identity accuracy.** Never trade correctness away for a benchmark win without an explicit new acceptance decision.
- [ ] **Step 5: Benchmark CPU first; test GPU only after CPU correctness.** NPU is optional and must not become a V0.2 requirement.
- [ ] **Step 6: If a candidate wins, replace packaged assets, update frozen contract JSON, rerun Android contract/inference tests and the Pixel acceptance suite. If none wins, keep float artifacts and document that decision.**
- [ ] **Step 7: Run full verification and commit:** `perf: finalize CardViper V0.2 model artifacts`.

## Phase B Completion Gate

Phase B is complete only when a green, stable-signed Android build on the Pixel 7 can select the held-out realistic blackjack images and, entirely offline, produce source-aligned boxes, exact rank+suit identities, explicit `BACK`, visible uncertainty, and correct KO/KISS III image-local deltas without changing the active shoe. The report must include whole-image correctness and failures, not only aggregate model accuracy.
