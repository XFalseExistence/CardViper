# Session A Phone Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a separate Android `sessionADemo` build that runs locally trained Session A detector and classifier artifacts through the existing full-table `IMAGE TEST` flow.

**Architecture:** A local-only Python pipeline prepares face-only Session A training records, trains a one-class FPNLite detector and a 53-output face-only classifier, then exports inspected LiteRT artifacts plus strict contracts. A dedicated Android build type loads those assets only when explicitly assembled, while normal debug/release variants retain unavailable production vision. The demo uses the existing detector/classifier adapters and labels the screen as Session-A, face-up-only.

**Tech Stack:** Python 3.11, TensorFlow 2.16 / pinned TensorFlow Object Detection API FPNLite-320, LiteRT, Kotlin, Android/Compose, Gradle build types.

**Spec:** `docs/superpowers/specs/2026-11-01-session-a-phone-demo-design.md`

## Global Constraints

- All inference is offline and runs from packaged LiteRT bytes; no source-image lookup or prerecorded answers.
- The Session A artifact is one family: training checks are smoke checks, never independent evaluation or published quality metrics.
- The demo supports face-up cards only; the `BACK` output has no training evidence and must be disclosed as unsupported.
- Normal debug/release production wiring remains `UnavailableCardDetector` plus `NoOpCardRecognizer`.
- No private images, model artifacts, contract bytes, checkpoints or generated training output may be tracked by Git.
- The detector has exactly one semantic class: `CARD`; classifier label order remains the canonical 53-label order, with `BACK` at index 52.
- `sessionADemo` remains an opt-in debug-signed build type with application ID suffix `.sessionademo`.

## Review Focus

- Missing/tampered local demo artifact: fail model initialization with `VisionModelUnavailableException`, not an empty card result. Covered by Task 5 asset-loader tests.
- Normal debug/release build: never enters Session A wiring even if source files are present. Covered by Task 5 build-mode tests.
- Face-down card in a Session A photo: show the persistent face-up-only warning; never claim BACK support or publish a BACK metric. Covered by Task 6 UI-mode tests.
- Detector output geometry after 320x320 resize: boxes map back to source-image pixels before crop extraction. Covered by Task 4 contract/adapter parity test.
- Low-confidence classifier result: it remains visibly uncertain under current Image Test behavior and never promotes to a shoe fact. Covered by Task 6 `ImageRecognitionViewModel` regression test.

---

## File structure

- `ml/cardviper_ml/session_a_demo_data.py`: creates explicit local demo manifests/crops from the audited Session A ZIP and approved group review.
- `ml/cardviper_ml/train_session_a_demo_classifier.py`: trains the face-only 53-logit classifier with mandatory demo metadata.
- `ml/cardviper_ml/train_session_a_demo_detector.py`: converts local records to TFRecord, configures and trains the one-class FPNLite detector.
- `ml/cardviper_ml/export_session_a_demo.py`: inspects actual LiteRT models, writes strict contracts and packages verified local assets.
- `ml/tests/test_session_a_demo_*.py`: pure, no-TensorFlow coverage of source guards, demo metadata, contracts and packaging.
- `app/src/main/java/com/cardviper/app/vision/litert/AndroidAssetLiteRtModelBundle.kt`: loads and hash-checks assets only from the APK asset manager.
- `app/src/main/java/com/cardviper/app/vision/litert/SessionADemoVisionFactory.kt`: creates real LiteRT adapters only for the demo build.
- `app/src/main/java/com/cardviper/app/CardViperApplication.kt`: chooses demo wiring solely through a build-type field.
- `app/src/main/java/com/cardviper/app/ui/imagetest/ImageTestScreen.kt`: adds a persistent demo-only face-up warning without changing result/ledger behavior.
- `app/build.gradle.kts`: declares the isolated build type and its asset packaging contract.
- `app/src/test/java/...`: asset-loader, factory and UI-mode tests.

### Task 1: Local Session A data preparation

**Files:**
- Create: `ml/cardviper_ml/session_a_demo_data.py`
- Create: `ml/tests/test_session_a_demo_data.py`
- Modify: `ml/pyproject.toml`

**Interfaces:**
- Consumes: local YOLOv8 archive, artifact audit, family review and output directory.
- Produces: `prepare_session_a_demo(archive: Path, audit: Path, reviewed_groups: Path, output: Path, *, seed: int) -> dict` and `cardviper-prepare-session-a-demo`.

- [ ] **Step 1: Write failing tests for exact source binding, approved review, 52 canonical face labels and fresh output enforcement**

```python
def test_prepare_rejects_unapproved_or_stale_review(tmp_path):
    with pytest.raises(ValueError, match="approved"):
        prepare_session_a_demo(archive, audit, review, tmp_path / "demo")
```

- [ ] **Step 2: Run the targeted test file**

Run: `PYTHONPATH=ml .venv-ml311/bin/python -m pytest ml/tests/test_session_a_demo_data.py -q`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement deterministic Session A manifest/crop preparation**

`prepare_session_a_demo` must verify the archive/review binding, refuse a non-approved review, write `demo_manifest.jsonl`, `train.jsonl`, `smoke.jsonl`, classifier crops and `demo_metadata.json`, and mark every report `SESSION_A_DEMO_ONLY` plus `NOT_INDEPENDENT`.

- [ ] **Step 4: Run the data preparation tests**

Run: `PYTHONPATH=ml .venv-ml311/bin/python -m pytest ml/tests/test_session_a_demo_data.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/cardviper_ml/session_a_demo_data.py ml/tests/test_session_a_demo_data.py ml/pyproject.toml
git commit -m "feat: prepare Session A demo data"
```

### Task 2: Face-only demo classifier training and export evidence

**Files:**
- Create: `ml/cardviper_ml/train_session_a_demo_classifier.py`
- Create: `ml/tests/test_train_session_a_demo_classifier.py`
- Modify: `ml/pyproject.toml`

**Interfaces:**
- Consumes: Task 1 manifests/crops and a fresh output directory.
- Produces: `train_session_a_demo_classifier(data_root: Path, output: Path, *, seed: int, epochs: int, batch_size: int) -> dict` and `cardviper-train-session-a-demo-classifier`.

- [ ] **Step 1: Write failing metadata tests**

```python
def test_demo_training_metadata_is_never_releasable(tmp_path):
    metadata = build_demo_metadata(...)
    assert metadata["usage"] == "SESSION_A_DEMO_ONLY"
    assert metadata["independent_evaluation"] is False
    assert metadata["supported_labels"] == list(LABELS[:-1])
```

- [ ] **Step 2: Run targeted tests**

Run: `PYTHONPATH=ml .venv-ml311/bin/python -m pytest ml/tests/test_train_session_a_demo_classifier.py -q`

Expected: FAIL because the demo trainer does not exist.

- [ ] **Step 3: Implement the explicit demo trainer**

Use the existing MobileNetV3Small architecture with 53 logits and canonical labels, fit only source crops whose labels are among the 52 faces, and write `BACK` as unsupported in metadata. Do not call the real B2 `require_ready` gate or reuse its metadata schema as evidence.

- [ ] **Step 4: Run the targeted tests**

Run: `PYTHONPATH=ml .venv-ml311/bin/python -m pytest ml/tests/test_train_session_a_demo_classifier.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/cardviper_ml/train_session_a_demo_classifier.py ml/tests/test_train_session_a_demo_classifier.py ml/pyproject.toml
git commit -m "feat: train face-only Session A demo classifier"
```

### Task 3: One-class Session A detector training

**Files:**
- Create: `ml/cardviper_ml/train_session_a_demo_detector.py`
- Create: `ml/tests/test_train_session_a_demo_detector.py`
- Modify: `ml/pyproject.toml`

**Interfaces:**
- Consumes: Task 1 image manifests and a pinned OD API checkout.
- Produces: `write_session_a_tfrecords(records: list[dict], root: Path) -> dict`, `train_session_a_demo_detector(...) -> dict`, and `cardviper-train-session-a-demo-detector`.

- [ ] **Step 1: Write failing pure-data tests for CARD-only TFRecord examples and pinned FPNLite config mutation**

```python
def test_detector_example_collapses_faces_to_card():
    example = detector_example(row_with_two_face_labels)
    assert decoded_classes(example) == [0, 0]
```

- [ ] **Step 2: Run targeted tests**

Run: `PYTHONPATH=ml .venv-ml311/bin/python -m pytest ml/tests/test_train_session_a_demo_detector.py -q`

Expected: FAIL because the detector trainer does not exist.

- [ ] **Step 3: Implement data conversion, FPNLite-320 training and LiteRT export**

Require the pinned OD API source used by the existing smoke tool, set `num_classes = 1`, consume only local Session A records, export a float LiteRT model with embedded postprocessing, and write `SESSION_A_DEMO_ONLY` metadata. Reject any model whose observed tensor surface is incompatible with Task 4's contract builder.

- [ ] **Step 4: Run targeted tests and the existing detector export smoke**

Run: `PYTHONPATH=ml .venv-detector/bin/python -m pytest ml/tests/test_train_session_a_demo_detector.py -q`

Run: `PYTHONPATH=ml .venv-detector/bin/python -m cardviper_ml.detector_export_smoke --odapi-root "$CARDVIPER_ODAPI_ROOT" --output /tmp/cardviper-detector-smoke`

Expected: tests PASS; smoke prints `INFERENCE_OK`.

- [ ] **Step 5: Commit**

```bash
git add ml/cardviper_ml/train_session_a_demo_detector.py ml/tests/test_train_session_a_demo_detector.py ml/pyproject.toml
git commit -m "feat: train Session A demo detector"
```

### Task 4: Verified Session A artifact packaging

**Files:**
- Create: `ml/cardviper_ml/export_session_a_demo.py`
- Create: `ml/tests/test_export_session_a_demo.py`
- Modify: `ml/pyproject.toml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Task 2/3 LiteRT artifacts and their metadata.
- Produces: `package_session_a_demo(classifier_dir: Path, detector_dir: Path, destination: Path) -> dict` and `cardviper-package-session-a-demo`.

- [ ] **Step 1: Write failing contract tests from fixture inspection reports**

```python
def test_package_rejects_detector_without_card_outputs(tmp_path):
    with pytest.raises(ValueError, match="detector"):
        package_session_a_demo(classifier, incompatible_detector, tmp_path / "assets")
```

- [ ] **Step 2: Run targeted tests**

Run: `PYTHONPATH=ml .venv-ml311/bin/python -m pytest ml/tests/test_export_session_a_demo.py -q`

Expected: FAIL because the packager does not exist.

- [ ] **Step 3: Implement artifact inspection and atomic local asset packaging**

Emit `detector.tflite`, `detector.json`, `classifier.tflite`, `classifier.json`, `labels.txt` and `manifest.json` into a fresh ignored asset directory. Contracts must declare the actual tensor surface, preprocessing, output names, source hashes and `SESSION_A_DEMO_ONLY`; the classifier contract remains canonical 53-output with `BACK` marked unsupported in the manifest.

- [ ] **Step 4: Run targeted tests**

Run: `PYTHONPATH=ml .venv-ml311/bin/python -m pytest ml/tests/test_export_session_a_demo.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml/cardviper_ml/export_session_a_demo.py ml/tests/test_export_session_a_demo.py ml/pyproject.toml .gitignore
git commit -m "feat: package verified Session A demo assets"
```

### Task 5: Android demo build type and fail-closed asset loader

**Files:**
- Modify: `app/build.gradle.kts`
- Create: `app/src/main/java/com/cardviper/app/vision/litert/AndroidAssetLiteRtModelBundle.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/litert/SessionADemoVisionFactory.kt`
- Modify: `app/src/main/java/com/cardviper/app/CardViperApplication.kt`
- Create: `app/src/test/java/com/cardviper/app/vision/litert/AndroidAssetLiteRtModelBundleTest.kt`
- Create: `app/src/test/java/com/cardviper/app/vision/litert/SessionADemoVisionFactoryTest.kt`

**Interfaces:**
- Consumes: Task 4 asset filenames/manifest and the existing `LiteRtCardDetector`/`LiteRtCardRecognizer` constructors.
- Produces: `AndroidAssetLiteRtModelBundle(context: Context, assetPrefix: String, kind: ModelKind): LiteRtModelBundleLoader` and `SessionADemoVisionFactory.create(context: Context): ImageRecognitionEngine`.

- [ ] **Step 1: Write failing tests for asset-byte hashing, missing assets and standard-build no-op wiring**

```kotlin
@Test fun missingModelAssetThrowsUnavailable() = assertFailsWith<VisionModelUnavailableException> {
    AndroidAssetLiteRtModelBundle(fakeAssets, "vision", ModelKind.DETECTOR).load()
}
```

- [ ] **Step 2: Run targeted JVM tests**

Run: `./gradlew test --tests '*AndroidAssetLiteRtModelBundleTest' --tests '*SessionADemoVisionFactoryTest'`

Expected: FAIL because demo wiring does not exist.

- [ ] **Step 3: Add the isolated `sessionADemo` build type and loader/factory**

Make `sessionADemo` inherit debug signing, use application ID suffix `.sessionademo`, and set exactly one `BuildConfig` boolean. Include `src/sessionADemo/assets/vision/` only in this build type. The loader validates the package manifest and every SHA-256 before opening LiteRT; normal debug/release continue to instantiate the current no-op/unavailable engine.

- [ ] **Step 4: Run targeted JVM tests and assemble both build types**

Run: `./gradlew test --tests '*AndroidAssetLiteRtModelBundleTest' --tests '*SessionADemoVisionFactoryTest' assembleDebug assembleSessionADemo`

Expected: PASS; both APKs assemble and only the demo APK contains the local assets.

- [ ] **Step 5: Commit**

```bash
git add app/build.gradle.kts app/src/main/java/com/cardviper/app/CardViperApplication.kt app/src/main/java/com/cardviper/app/vision/litert app/src/test/java/com/cardviper/app/vision/litert
git commit -m "feat: add fail-closed Session A demo wiring"
```

### Task 6: Demo disclosure and Image Test regression coverage

**Files:**
- Modify: `app/src/main/java/com/cardviper/app/ui/imagetest/ImageTestScreen.kt`
- Modify: `app/src/test/java/com/cardviper/app/ui/imagetest/ImageRecognitionViewModelTest.kt`
- Create: `app/src/test/java/com/cardviper/app/ui/imagetest/SessionADemoDisclosureTest.kt`
- Modify: `README.md`

**Interfaces:**
- Consumes: the `BuildConfig.SESSION_A_DEMO` mode established by Task 5.
- Produces: persistent `SESSION A / FACE-UP DEMO` disclosure in the demo build and unchanged ordinary Image Test copy in standard builds.

- [ ] **Step 1: Write failing mode/copy and low-confidence regression tests**

```kotlin
@Test fun demoDisclosureStatesFaceUpOnlyAndOffline() {
    assertTrue(demoDisclosure.contains("FACE-UP"))
    assertTrue(demoDisclosure.contains("OFFLINE"))
}
```

- [ ] **Step 2: Run targeted JVM tests**

Run: `./gradlew test --tests '*SessionADemoDisclosureTest' --tests '*ImageRecognitionViewModelTest'`

Expected: FAIL because the demo disclosure is absent.

- [ ] **Step 3: Add demo-only disclosure and document installation boundaries**

Keep `IMAGE TEST` as the workflow. The disclosure must state local processing, Session A limitation, face-up-only restriction, no ledger changes and no quality claim. Do not add live inference or alter count strategy behavior.

- [ ] **Step 4: Run targeted tests**

Run: `./gradlew test --tests '*SessionADemoDisclosureTest' --tests '*ImageRecognitionViewModelTest'`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/cardviper/app/ui/imagetest/ImageTestScreen.kt app/src/test/java/com/cardviper/app/ui/imagetest README.md
git commit -m "feat: disclose Session A demo boundaries"
```

### Task 7: End-to-end local handoff and device verification

**Files:**
- Create: `ml/reports/session-a-phone-demo.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1–6, local archive and approved group review.
- Produces: exact Mac commands for local model creation, packaging, demo APK assembly, installation and device checks.

- [ ] **Step 1: Add an executable-order handoff document**

The document must require a Python 3.11 environment, private local paths, fresh output directories, actual artifact inspection, local asset copy, `assembleSessionADemo`, APK signature comparison, 16 KB emulator installation and S22+ installation. It must label every output `SESSION_A_DEMO_ONLY` and prohibit promotion.

- [ ] **Step 2: Validate documentation paths and ignored outputs**

Run: `git check-ignore ml/local/session-a-demo app/src/sessionADemo/assets/vision/detector.tflite`

Expected: both local model paths are ignored.

- [ ] **Step 3: Run available verification**

Run: `PYTHONPATH=ml .venv-ml311/bin/python -m pytest ml/tests -q`

Run: `./gradlew test lintDebug assembleDebug assembleSessionADemo`

Run: `ANDROID_SERIAL="$(adb -e get-serialno)" ./gradlew connectedSessionADemoAndroidTest`

Expected: all runnable checks PASS; any hosted-environment dependency-resolution limitation is documented separately from Mac evidence.

- [ ] **Step 4: Commit**

```bash
git add ml/reports/session-a-phone-demo.md README.md
git commit -m "docs: hand off Session A phone demo workflow"
```

## Self-review

- **Spec coverage:** Tasks 1–4 build and bind real local artifacts; Tasks 5–6 isolate Android wiring and user disclosure; Task 7 supplies the Mac/device path. The production lock, offline requirement, Session A limitation, no ledger writes, one-class detector and unsupported BACK condition are covered.
- **Step scan:** Every task begins with a named failing test, has a narrow implementation surface, then verifies and commits.
- **Type consistency:** Task 4 writes the filenames/manifest consumed by Task 5. Task 5 supplies build mode consumed by Task 6. Task 1 output supplies Tasks 2 and 3.
- **Review focus:** Each listed failure mode has a named owning task/test.
- **Proportion:** The plan names precise files and interfaces without prescribing implementation bodies.
