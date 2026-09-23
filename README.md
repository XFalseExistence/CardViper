# CardViper

CardViper is an offline Android blackjack computer-vision training platform.

## Current V0.1

CardViper currently includes:

- KO as the default count system
- KISS III as an alternate mode
- persistent shoe/session state
- append-only card ledger with correction/invalidation replay
- CameraX live rear-camera preview
- manual card entry and undo
- recent-card correction/removal
- pending-review plumbing and hard-case metadata capture
- Room persistence and DataStore preferences
- GitHub Actions APK builds

The neural card-recognition model is not enabled yet. The live camera currently provides the capture/HUD foundation for the next vision milestone.

## Architecture

`Camera -> Vision -> Tracking -> Ledger -> Blackjack Brain -> HUD`

Vision and tracking report physical-card facts. The ledger persists those facts. KO and KISS III interpret the same resolved ledger, so changing count strategy does not rewrite shoe history.

## Android target

Primary development/test device: Google Pixel 7 on Android 16.

Package: `com.cardviper.app`

Minimum Android SDK: 26

## Download a debug APK on a phone

1. Open this repository on GitHub while signed in.
2. Open **Actions**.
3. Open the latest successful **Android Build** run on `main`.
4. Scroll to **Artifacts**.
5. Tap **CardViper-debug-apk**.
6. Open the downloaded ZIP.
7. Install `app-debug.apk`.

Android may ask you to allow installs from the browser or file manager you used.

## Debug signing

GitHub Actions debug APKs use a stable public AOSP test certificate so successive CI builds can update one another in place. This signer is intentionally **debug-only** and must never be used for a production/release build.

If you installed a CardViper APK from before the stable signer was introduced, Android will reject the first stable-signed update because the old APK used a different debug certificate. Uninstall that older CardViper build once, install the current CI APK, and future CI APKs can update it normally.

## Cloud verification

Every main-branch build runs:

```bash
./gradlew test
./gradlew assembleDebugAndroidTest
./gradlew lintDebug
./gradlew assembleDebug
```

The workflow also verifies the APK signer and uploads `app-debug.apk` as artifact `CardViper-debug-apk`.

## Roadmap

- V0.1 — Android foundation, count engines, persistence, HUD, CameraX, correction/review
- V0.2 — image-file card recognition test mode
- V0.3 — recorded-video recognition/replay
- V0.4 — live card recognition on Pixel 7
- V0.5 — temporal physical-card tracking and duplicate suppression
- V0.6 — dynamic FPV ROI and performance/thermal tuning
- V0.7 — shoe/penetration intelligence and expanded training statistics

## Offline policy

The installed CardViper app does not require a cloud backend, account, analytics, ads, or network service for normal operation.
