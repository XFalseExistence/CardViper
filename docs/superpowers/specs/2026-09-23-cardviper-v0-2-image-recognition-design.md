# CardViper V0.2 Image Recognition Design

## Goal
V0.2 gives CardViper its first real computer-vision capability: select a blackjack-table image containing multiple visible cards, detect every card, identify each face-up card by full rank and suit, recognize face-down cards explicitly, and display both a labeled image overlay and count-preview output.

This milestone proves the same detector/recognizer architecture that will later power recorded-video and live CameraX recognition. V0.2 remains an offline diagnostic image workflow; it does not yet perform continuous tracking, live haptic decisions, or automatic shoe-ledger commits.

## Product intent and constraints
- Primary development/test device: Google Pixel 7.
- Fully offline. No cloud inference, accounts, network dependency, or telemetry.
- Input is a user-selected still image from Android's photo picker.
- Input may contain multiple cards in a realistic blackjack-table scene.
- Primary future geometry is chest-level / first-person, with whole-table context and relatively small cards.
- Output must include both:
  - image overlay with a box, predicted identity, and confidence for each detected card;
  - interpreted card list plus KO and KISS III count deltas.
- Exact face identity is required: all 52 rank/suit combinations.
- Face-down cards are a first-class `BACK` identity, making the recognizer effectively 53-way.
- Low-confidence face-up detections still produce a best-guess identity and contribute to the V0.2 diagnostic count preview, but are visibly marked uncertain.
- Low-confidence observations must not be treated as haptic-safe or committed facts in future live use until temporal confirmation.

## Scope boundary
V0.2 includes:
- image picker flow;
- multi-card detection;
- high-resolution per-card crops;
- crop normalization;
- 53-way card classification;
- confidence handling;
- overlay rendering;
- interpreted-card list;
- KO/KISS III count preview;
- deterministic fake-model plumbing for CI;
- real-model inference on device.

V0.2 deliberately excludes:
- continuous CameraX inference;
- physical-card multi-object tracking;
- temporal consensus across frames;
- haptic alerts;
- automatic ledger commits;
- automatic shuffle/reset detection;
- four-corner/keypoint perspective model unless padded-crop classification proves insufficient during Pixel testing.

## Architecture
The V0.2 still-image pipeline is:

`Photo Picker -> Decode full image -> Wide card detector -> Source-resolution padded crops -> Normalize crop -> 53-way classifier -> Confidence policy -> Overlay + interpreted results -> KO/KISS preview`

The detector answers **where cards are**. The classifier answers **which card each crop represents**.

Vision does not directly mutate count state. It emits observations. The diagnostic preview passes interpreted observations through existing count-strategy logic without writing them to the active shoe ledger.

### Main components

#### ImageTestScreen
Dedicated UI flow reachable from CardViper navigation as `IMAGE TEST`.

Responsibilities:
- launch Android photo picker;
- display the selected image;
- start/cancel inference;
- render labeled detection overlay;
- render result list and count preview;
- surface stage-specific errors;
- make uncertainty visible.

#### ImageRecognitionViewModel
Owns the still-image inference state machine.

Suggested states:
- IDLE
- LOADING_IMAGE
- DETECTING
- CLASSIFYING
- COMPLETE
- ERROR

Responsibilities:
- coordinate image decode and inference off the main thread;
- preserve immutable recognition results for the UI;
- calculate preview count deltas through existing `CountStrategy` implementations;
- never write observations into the shoe ledger in V0.2.

#### CardDetector
Use the existing vision boundary rather than introducing a UI-specific detector API.

Input:
- decoded image / inference tensor representation.

Output per candidate:
- bounding rectangle in source-image coordinates;
- detection confidence.

The detector has one semantic object class: `CARD`. It does not decide rank or suit.

#### CardRecognizer
Receives one normalized card crop and returns a classification distribution.

V0.2 label space:
- 52 exact face identities;
- `BACK`.

Output includes:
- best identity;
- best-class confidence;
- optional top-N candidates for diagnostics;
- uncertainty flag derived from policy, not hard-coded into the model.

#### ImageRecognitionEngine
A small orchestration layer that keeps detector, crop extraction, classifier, and result assembly independent from Compose.

Responsibilities:
1. run detector;
2. clamp/pad candidate boxes safely inside source bounds;
3. crop from the original decoded image, not the detector's downscaled tensor;
4. normalize each crop for the classifier;
5. classify each candidate;
6. return image-coordinate observations in deterministic order.

This layer is reusable by later video/live pipelines.

## Model/runtime strategy
Use two independently replaceable mobile models:

1. **Wide card detector** — optimized to locate multiple cards in full blackjack scenes at different sizes, rotations, lighting conditions, and partial occlusions.
2. **53-way classifier** — optimized for a tight card crop and returns one of 52 exact identities or `BACK`.

Models are packaged with the app and executed entirely on device through an Android mobile inference runtime. The implementation must preserve `CardDetector` and `CardRecognizer` interfaces so the concrete model format/runtime can be upgraded without changing UI, ledger, or count-engine code.

Initial implementation should use a LiteRT / TensorFlow-Lite-compatible on-device model format and begin with the most broadly reliable device execution path. Delegate acceleration may be added only after correctness is established on the Pixel 7.

No model may require a network call at initialization or inference time.

## Crop and geometry strategy
V0.2 intentionally begins without a separate four-corner/keypoint model.

For each detected card:
- expand the detector rectangle by a small bounded padding factor;
- clamp to image bounds;
- crop from the original source-resolution bitmap;
- normalize to the classifier input size while preserving enough surrounding edge information for orientation/context;
- apply classifier orientation normalization if required by the trained model.

Strong perspective distortion is expected to lower confidence initially. If Pixel acceptance images show systematic failures caused by perspective rather than classification quality, the next refinement is explicit four-corner detection + projective rectification behind the same crop/recognizer boundary.

## Observation model
A still-image result should preserve at least:
- source bounding box;
- detector confidence;
- predicted identity;
- classifier confidence;
- uncertainty flag;
- optional top-N alternatives;
- face state derived from identity (`BACK` -> FACE_DOWN, otherwise FACE_UP).

A V0.2 observation is diagnostic and ephemeral. It is not a `CARD_COMMITTED` event and has no persistent `trackId` requirement.

## Confidence policy
V0.2 must not hide uncertainty.

For a face-up crop:
- always return the classifier's best identity when inference succeeds;
- if confidence meets the configured confident threshold, render as normal/confident;
- if confidence is below threshold, render the same best guess in yellow and mark it uncertain;
- include the best guess in the diagnostic KO/KISS preview.

For `BACK`:
- render `BACK` visibly;
- contribute zero to both count previews;
- do not invent a hidden face identity.

This V0.2 preview rule does **not** change the live-system safety rule. In future video/live operation, low-confidence single-frame observations remain provisional and may not trigger haptics or become permanent ledger facts until temporal confirmation.

## Count preview behavior
The image test screen calculates two independent deltas from detected face-up identities:

### KO preview
Use the existing KO strategy tags:
- 2-7: +1
- 8-9: 0
- 10/J/Q/K/A: -1

### KISS III preview
Use the existing color-aware strategy:
- black 2: +1
- red 2: 0
- 3-7: +1
- 8-9: 0
- 10/J/Q/K/A: -1

Because V0.2 predicts exact suit, KISS III has the color information it needs.

`BACK` contributes zero.

The screen reports image-local deltas only. It must not alter the current shoe running count or append ledger events.

## UI design
Add `IMAGE TEST` as a dedicated route from the app's test/development-facing navigation.

### Empty state
- `CHOOSE IMAGE` action;
- short offline/local-processing note;
- no placeholder detections.

### Processing state
Show the selected image and a concise stage indicator such as:
- `DETECTING CARDS`
- `CLASSIFYING 3/7`

Do not freeze the Compose UI while inference runs.

### Complete state
Image overlay:
- bounding box per card;
- identity label such as `7♦` or `K♣`;
- confidence percentage when useful;
- `BACK` for face-down cards;
- uncertain predictions rendered yellow.

Result panel:
- detected-card count;
- ordered interpreted list;
- per-card confidence/uncertain state;
- KO delta;
- KISS III delta;
- total uncertain count.

No result on this screen is automatically committed to the active shoe.

## Error handling
Failures must identify the stage rather than collapsing into a generic message.

Required cases:
- image URI unavailable or permission revoked -> image-load error;
- decode failure -> image-decode error;
- model unavailable/incompatible -> model-load error;
- detector inference failure -> detector error;
- crop extraction failure for one box -> skip/flag that candidate without crashing the entire image when safe;
- classifier inference failure -> classifier error;
- zero candidates -> `NO CARDS DETECTED`, not a crash or generic error.

The UI must remain recoverable: the user can choose another image after any failure.

## Training strategy
### Detector dataset
Train/evaluate on full-scene images containing multiple cards and realistic distractors.

Important variation:
- blackjack felt and other table surfaces;
- dealer/player hands;
- chips and chip stacks;
- card trays/discard areas;
- multiple card scales;
- arbitrary rotations;
- partial occlusion;
- glare;
- hard shadows;
- motion/compression blur;
- chest-level / first-person perspective;
- overhead-ish reference views;
- images containing no cards.

### Classifier dataset
Train on tight card crops covering all 52 identities plus varied `BACK` examples.

Augmentation should include:
- rotation;
- scale reduction;
- mild perspective distortion;
- exposure/contrast shifts;
- blur;
- compression artifacts;
- glare simulation;
- partial edge/corner occlusion;
- realistic crop padding variation.

### Data leakage rule
Validation/test splits are by scene/source/session, not random near-duplicate frames. Frames from the same captured sequence must not be split across train and evaluation sets in a way that inflates accuracy.

### FPV hard-case set
Maintain a small Pixel-7-oriented chest-level evaluation set that is not initially used for training. It acts as the practical reality check before advancing to video recognition.

## Evaluation metrics
Do not rely on a single aggregate accuracy number.

Track at least:
- card detection recall;
- detector false positives per image;
- exact 52-face identity accuracy on detected face-up cards;
- `BACK` accuracy;
- uncertainty rate;
- end-to-end whole-image correctness: percentage of evaluation images where every visible card is detected and identified correctly with no extra false card.

Whole-image correctness is the most operationally meaningful V0.2 metric because one wrong identity can alter the count preview even when per-card accuracy appears high.

## Deterministic test seam
Keep real inference behind interfaces and provide deterministic fake implementations.

CI tests should be able to inject:
- known detector boxes;
- known classifier outputs;
- `BACK` results;
- low-confidence results;
- detector/classifier failures.

This allows UI/domain plumbing to be tested without neural models or device-specific acceleration.

## Automated verification
Tests should cover at minimum:
- multiple detector candidates preserve source coordinates;
- padded crop bounds clamp correctly at all four image edges;
- classifier identity maps correctly to `PlayingCard` rank/suit/color;
- `BACK` maps to face-down and zero count delta;
- low-confidence face prediction remains present, is marked uncertain, and contributes to diagnostic preview;
- exact-suit 2 drives correct KISS III red/black behavior;
- KO and KISS deltas use existing strategy code rather than duplicate tag tables in UI;
- no image-test result writes to Room ledger/session state;
- no-card detector result produces `NO CARDS DETECTED` state;
- stage-specific errors are recoverable;
- fake engine produces deterministic overlay/result state.

Existing CI remains authoritative:
- `./gradlew test`
- `./gradlew assembleDebugAndroidTest`
- `./gradlew lintDebug`
- `./gradlew assembleDebug`

## Pixel 7 physical acceptance set
Use a small repeatable group of real images including:
- clean overhead-ish multi-card table image;
- chest-level FPV image;
- rotated cards;
- partially occluded cards;
- glare/shadow case;
- multiple cards sharing the same rank;
- at least one face-down card;
- no-card distractor image.

For each applicable image verify:
- every intended card has the correct box;
- no spurious card boxes;
- every face-up card has the correct exact identity;
- face-down card reports `BACK`;
- uncertainty is visibly flagged where expected;
- KO delta is correct;
- KISS III delta is correct;
- choosing another image and rerunning remains stable.

## Relationship to future live haptics
V0.2 does not emit haptics, but it must preserve the correct boundary for them.

Future flow:
`frame observations -> temporal consensus -> physical tracker -> confirmed ledger state -> count/strategy decision -> haptic output`

A low-confidence single-frame guess may exist provisionally, but it must not trigger haptic output. Haptics fire only from confidence/temporal-confirmed state.

The haptic transport remains hardware-agnostic: Android phone vibration first, future watch/LRA output later, without changing counting or recognition logic.

## V0.2 success condition
On the Pixel 7, the user can open `IMAGE TEST`, choose a realistic blackjack-table photo containing multiple cards, run inference fully offline, and receive:
- a bounding box for each detected card;
- exact rank+suit labels for face-up cards;
- explicit `BACK` labels for face-down cards;
- visible uncertainty on weak predictions;
- an interpreted card list;
- correct KO and KISS III image-local count deltas.

The image-test workflow must remain separate from the active shoe ledger, and failures must be diagnosable by stage. The architecture must be reusable for the subsequent video/live milestone without replacing the detector/recognizer contracts.