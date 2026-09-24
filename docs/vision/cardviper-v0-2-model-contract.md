# CardViper V0.2 Model Contract

Phase A provides Android image-test plumbing and pure Kotlin adapter boundaries. Real trained models and on-device runtime adapters are Phase B deliverables; they are not integrated yet.

## Required assets

- Card detector model: multi-card, one semantic `CARD` class.
- Card classifier model: 53 outputs matching `CardLabelCodec`.
- Label file: exactly 53 canonical labels, one per line, with no duplicates or missing identities.

Asset filenames and model-specific tensor contracts are not assigned in Phase A.

## Required detector adapter semantics

Implement `CardDetector.detect(image: VisionImage): List<CardCandidate>` as a suspend function. Return `CardCandidate(x, y, width, height, confidence)` boxes in **source-image pixel coordinates**, with `x`/`y` at the top-left, plus detector confidence. Coordinates and dimensions must be finite; confidence must be finite and in `0..1`.

Here, source means the supplied, bounded decoded `VisionImage`, not the original URI's dimensions or a downscaled detector tensor. The adapter must reverse any model resize/letterbox transform before returning boxes. The model's native box format remains to be established from its export.

Finite boxes may cross image edges or have nonpositive area. `DefaultImageRecognitionEngine` sorts by `y`, then `x`, then descending detector confidence. It pads/clamps boxes and skips degenerate or empty crops. A successful detector may return an empty list; an unavailable model must throw `VisionModelUnavailableException`, not pretend that no cards were detected.

## Required classifier adapter semantics

Implement `CardRecognizer.recognize(crop: VisionImage): CardRecognition` as a suspend function. Return:

- One of 52 exact rank/suit identities, `CardIdentity.Face(PlayingCard(rank, suit))`, or the separate `CardIdentity.Back` identity.
- Classifier confidence, finite and in `0..1`.
- Optional top-N alternatives as `List<RankedIdentity>`, each with an identity and finite `0..1` confidence.

`BACK` is not a `PlayingCard`; do not invent a hidden rank or suit. Exact suit determines card color, including the KISS III distinction between red and black twos.

The input is a source-resolution crop copied from the supplied decoded image. `VisionImage` contains width, height, and row-major packed RGB bytes: three bytes per pixel, no row padding. These are domain image bytes, **not a model input dtype, tensor layout, or normalization contract**. Model-specific resize, orientation handling, normalization, tensor packing and quantization belong inside the adapter after the real export contract is known.

The current engine uses configurable defaults of 10% crop padding per side and a `0.80` classifier confidence threshold. It retains best guesses below the threshold and marks them uncertain. These values are application policy, not inferred model thresholds. `BACK` contributes zero; other identities go through the existing `KoStrategy` and `Kiss3Strategy` for image-local preview deltas. No result writes to the shoe ledger or triggers haptics.

Preserve coroutine cancellation and distinguish unavailable models from inference failures. The engine preserves `VisionModelUnavailableException` and wraps unexpected detector/classifier failures in `DetectorInferenceException`/`ClassifierInferenceException` respectively. Do not silently substitute fake predictions for failed inference.

## Frozen label source of truth

[`CardLabelCodec`](../../app/src/main/java/com/cardviper/app/vision/CardLabelCodec.kt) is authoritative. Canonical labels are case-sensitive ASCII:

- Rank token: `A`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `J`, `Q`, `K`.
- Suit token: `C`, `D`, `H`, `S`.
- Face label: rank token followed by suit token, such as `AC`, `10H` or `KS`.
- Face-down label: exactly `BACK`.

Phase B export tests must prove that the packaged label file contains **exactly the codec's full 53-label set**, with 53 unique entries and successful decode/encode round trips. Generate the expected face set from every `CardRank`/`CardSuit` pair through the codec, then include `BACK`.

Set equality alone does not establish output order. A separate test must prove each exported classifier output index maps to the correct label-file entry. Neither the lists above nor Kotlin enum/map iteration order defines model output order.

## Fields Phase B must freeze from real exported models

- Detector input width/height/channels.
- Detector input dtype.
- Detector input quantization.
- Detector normalization.
- Detector tensor names or indices.
- Detector box format.
- Detector score tensor.
- Detector class tensor.
- NMS ownership and thresholds.
- Classifier input width/height/channels.
- Classifier input dtype.
- Classifier quantization.
- Classifier normalization.
- Classifier output dtype/quantization.
- Classifier output ordering.
- Classifier output-to-label mapping.

**NONE of those model-specific tensor details are guessed in Phase A.** They must come from the actual exported detector/classifier artifacts and be covered by tests. This includes documenting tensor layout, quantization parameters and whether postprocessing is inside the exported model or the adapter, as applicable to those artifacts.

Adapters must initialize and infer entirely offline using packaged assets. Current production wiring intentionally uses `UnavailableCardDetector` and `NoOpCardRecognizer`; selecting a decodable image reports `VISION MODEL NOT INSTALLED`. Replace that wiring only when Phase B supplies real, tested adapters and assets. Validate them against the [Pixel 7 hard-case acceptance set](../superpowers/specs/2026-09-23-cardviper-v0-2-image-recognition-design.md#pixel-7-physical-acceptance-set); Phase A host tests are not evidence of neural recognition accuracy.
