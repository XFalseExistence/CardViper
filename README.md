# CardViper

CardViper is a blackjack computer-vision training platform. The Android app works offline.

## Current milestone: V0.2 Phase A

Phase A provides the image-test pipeline and diagnostic UI:

- Android Photo Picker and bounded image decoding: aspect-preserving, no upscaling, maximum 4096-pixel long edge.
- Exact 52 rank/suit identities plus the separate `BACK` identity.
- Multi-card detector/classifier orchestration behind replaceable interfaces.
- Source-resolution padded/clamped crops from the decoded image.
- Confidence, uncertainty, progress, cancellation and recoverable error handling.
- KO/KISS III image-local count previews using the existing count strategies.
- **IMAGE TEST** screen with fitted-image overlays, ordered results and uncertainty markers.
- Session isolation: image analysis never changes the active shoe or its ledger.
- Stable debug build/signing and downloadable APKs through GitHub Actions.

**Real trained detector/classifier assets are NOT yet integrated.** Production intentionally uses `UnavailableCardDetector` with `NoOpCardRecognizer`. Selecting a decodable image in **IMAGE TEST** therefore reports **VISION MODEL NOT INSTALLED** until Phase B. Automatic card recognition is not complete; deterministic fake implementations exercise the pipeline in tests, not in production.

Intended runtime path:

```text
Photo Picker -> detector -> source-resolution crop -> 53-way classifier -> overlay + KO/KISS preview
```

Bounded decoding precedes detection. The displayed image and crops use the same decoded geometry; model-specific preprocessing belongs behind the adapters. See the [V0.2 model handoff contract](docs/vision/cardviper-v0-2-model-contract.md).

## Existing shoe workflow

V0.1 features remain available: KO by default, KISS III alternate mode, CameraX rear-camera preview, manual entry/undo, recent-card corrections/invalidation, pending-review and hard-case metadata plumbing, Room persistence, and DataStore preferences. KISS III currently supports the six-deck profile only.

The append-only ledger preserves card facts; both counting strategies replay the same resolved history. IMAGE TEST is a separate diagnostic path and never commits its predictions to that history. Live camera recognition is not enabled.

## Android target

Google Pixel 7 on Android 16; package `com.cardviper.app`. Compile SDK 37, target SDK 36, minimum SDK 26, Java 17.

## Download a debug APK on a phone

1. Open this repository on GitHub while signed in.
2. Go to **Actions -> Android Build -> latest successful run on main**.
3. Under **Artifacts**, download **CardViper-debug-apk**.
4. Extract the ZIP and install `app-debug.apk`.

Android may ask you to allow installs from the browser or file manager you used.

## Debug signing

GitHub Actions debug APKs use a stable public AOSP test certificate so successive CI builds can update one another in place. This signer is intentionally **debug-only** and must never be used for production/release builds.

An APK installed before the stable signer was introduced may have a different certificate. Android rejects updates across certificates; replacing that older installation requires uninstalling it first, which removes its local app data. Subsequent stable-signed CI APKs can update normally.

## Verification

Run locally with Java 17 and the Android SDK, or use the existing **Android Build** workflow:

```bash
./gradlew test
./gradlew assembleDebugAndroidTest
./gradlew lintDebug
./gradlew assembleDebug
```

CI also verifies the APK signer and uploads `app-debug.apk` as **CardViper-debug-apk**. `assembleDebugAndroidTest` compiles/packages instrumented tests; it does not run them on a device. Pixel 7 Photo Picker/rendering, image memory use and real recognition acceptance require physical verification.

## Next: Phase B models and runtime adapters

1. Acquire/prepare datasets, keeping evaluation scenes separate from training sources.
2. Train the multi-card detector and 53-class classifier.
3. Export both models and freeze their **real** tensor contracts and output-to-label mappings in tests.
4. Implement standalone offline on-device runtime adapters behind `CardDetector` and `CardRecognizer`.
5. Package model assets and replace the `UnavailableCardDetector` production wiring with tested adapters.
6. Run the Pixel 7 hard-case acceptance set: chest-level FPV, rotations, occlusion, glare/shadow, repeated ranks, face-down cards and no-card scenes.

Phase B has not started. No model tensor shapes, quantization, normalization or output ordering are guessed by Phase A.

Later milestones remain recorded-video replay, live recognition, temporal tracking/duplicate suppression, FPV performance tuning, and expanded training statistics.

## Offline policy

No cloud backend, accounts, analytics, ads or app networking. Camera is the only app-facing permission needed for live preview; Photo Picker requires no broad storage/media permission. The manifest removes the network-state permission inherited from Media3. Normal build tooling may download dependencies and the public debug signing material.
