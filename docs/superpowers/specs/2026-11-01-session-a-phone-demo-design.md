# CardViper Session A phone demo design

## Intent

Deliver a temporary Android build that performs real, fully offline inference on
full table images on William's Samsung S22+. It is intended to demonstrate the
complete path—photo picker, card detection, per-card classification, overlays
and KO/KISS image-local previews—using the approved Session A DroidCam export.

The build exists to make the model path visible on the phone. It is not a
baseline, release model, accuracy claim or replacement for the B2/B3 evidence
gates.

## What users will see

The existing `IMAGE TEST` flow remains the entry point. In the demo build it:

1. accepts a still table image through the Android photo picker;
2. finds face-up playing cards locally;
3. labels each found card with rank, suit and confidence;
4. draws the existing image overlays and image-local KO/KISS previews; and
5. keeps every result outside the shoe ledger.

The screen visibly identifies the build as `SESSION A / FACE-UP DEMO`. It
states that it is optimized for the Session A DroidCam scene and that card
backs, new environments and live camera inference are unsupported.

## Constraints and non-goals

- Inference is real LiteRT inference on the phone. There are no image hashes,
  prerecorded annotations, source-image lookups or fabricated predictions.
- The session contains the 52 face identities only. A face-down `BACK` class is
  deliberately unavailable rather than synthesized or silently guessed.
- All source frames form one family. Any development-only split may be used for
  training progress, but no metric is described as independent validation or
  test accuracy.
- The normal production build keeps `UnavailableCardDetector` and
  `NoOpCardRecognizer`. No Session A model, private image, model binary or
  generated asset is tracked or published with the production variant.
- The demo neither enables live CameraX recognition nor writes to the counting
  ledger.

## Approaches considered

### 1. Pack source annotations or image matches

Rejected. It would draw impressive overlays but would not be recognition.

### 2. Single-card classifier screen

Useful as a faster proof, but it does not satisfy the requested full-table
phone demonstration because it has no card-location stage.

### 3. Isolated full-table Session A demo

Selected. Train one `CARD` detector from the Session A full-image annotations
and a 52-face classifier from its annotated crops. Export both as LiteRT and
run them through the existing photo-picker pipeline in a demo-only Android
variant. This proves the complete on-device inference route while keeping its
limits explicit.

## Data and training design

### Detector

- Convert the local YOLO annotations to a one-class `CARD` record format.
- Use the pinned TensorFlow Object Detection API FPNLite-320 export family that
  the repository's existing detector smoke validates.
- Train only from the local Session A artifact; reserve a deterministic temporal
  slice solely for smoke checking during the demo build.
- Record artifact hashes, source archive hash, configuration and the fact that
  all data is one session in untracked demo metadata.

### Classifier

- Derive source-resolution crops from the same annotations and train exactly
  the 52 canonical face labels.
- It is a face-only demo recognizer. It cannot emit `BACK` and must not be
  substituted for the production 53-way recognizer.
- Perform a deterministic holdback from Session A only to catch malformed
  exports or unusable inference; do not publish it as a quality metric.

### Demo asset contract

- Generate an explicit detector contract from the actual exported LiteRT
  tensor names, shapes, preprocessing and postprocessing semantics.
- Generate an explicit 52-face demo-classifier contract from the actual
  artifact. It is distinct from the production 53-way contract.
- Verify SHA-256 values before Android loads either model.
- Store artifacts under ignored `ml/local/session-a-demo/`; an opt-in local
  copy task places them only into the demo variant's generated assets.

## Android design

### Isolation

- Introduce a build variant dedicated to `sessionADemo`; it is not the
  release or ordinary debug variant.
- Variant-specific composition injects concrete LiteRT detector and demo
  recognizer only when both local verified assets are present.
- The default app configuration remains unchanged and model-unavailable when
  production assets do not exist.

### Face-only recognizer boundary

The existing `LiteRtCardRecognizer` correctly enforces a production 53-way
contract. The demo therefore receives a separate, visibly named recognizer
implementation and contract parser that accept exactly the 52 canonical face
labels. It converts only those labels to `CardIdentity.Face` and has no code
path that turns an unknown or face-down card into a face prediction.

The demo screen warns the user before inference that every visible card must be
face-up. A detector crop with no confident face classification is rendered as
`UNSURE`, contributes nothing to image-local count previews and never acquires
a made-up card identity.

## Validation

Automated checks add coverage for:

- strict 52-face demo labels and rejection of `BACK`/duplicate/missing labels;
- missing or tampered demo assets fail closed;
- source-image detector coordinates remain correct through the LiteRT adapter;
- low-confidence/unsupported crops render as uncertain and have no count delta;
- demo wiring is impossible in the ordinary debug/release build;
- the current 53-way production contract tests remain unchanged.

Before handing over an APK, run the Android unit tests, assemble the demo APK,
validate its packaged asset hashes, run the existing 16 KB emulator test, and
install the demo only as a debug build on the S22+ after signer compatibility
is checked.

## User workflow

1. Build the local Session A artifacts on the Mac from the private archive.
2. Verify the generated metadata says `SESSION_A_DEMO_ONLY` and `FACE_UP_ONLY`.
3. Assemble and install the isolated demo APK.
4. Open `IMAGE TEST`, choose a Session-A-style full-table image, and inspect
   local overlays/results.

## Promotion rule

Nothing from this variant can be promoted merely because it works on the
Session A scene. Promotion requires the ordinary B2/B3 flow: independently
grouped face and BACK data, detector negatives, valid held-out evaluation,
53-way classifier evidence, real detector evidence, and physical acceptance.
