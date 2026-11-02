# Session A phone-demo handoff

This is an intentionally isolated **`SESSION_A_DEMO_ONLY`** workflow. It creates
a debug-signed `sessionADemo` APK which runs LiteRT models locally on a phone.
It is limited to the approved single DroidCam Session A source, face-up cards
and still images. It is not independent evaluation, a quality claim, a
production model, live-camera recognition or a shoe-ledger input.

Do not commit any output below. The training workspace stays on the external
disk, and the repository ignores `app/src/sessionADemo/assets/vision/`.

## Prerequisites

- Run from the CardViper checkout with its Python 3.11 environment at
  `.venv-ml311/bin/python`.
- Retain the approved Session A archive, artifact audit and reviewed-groups
  JSON on the local external disk. The reviewed-groups file must be the
  `READY` output from `cardviper_ml.apply_family_review`, not the original
  review-decision form.
- Supply the already-pinned local TensorFlow Object Detection API checkout and
  a local OD API-compatible FPNLite checkpoint prefix. All private data and
  the detector checkpoint stay local; the finished app never downloads models.
- Start `CardViper_16K_Test` and connect the S22+ only after checks below.

## Build local artifacts

Use fresh paths; each command refuses to overwrite an existing output.

```zsh
set -euo pipefail

cd "$HOME/.codex/.chatgpt-projects/g-p-6a67818aff288191b9df38d4124bdcd2/CardViper"

PY="$PWD/.venv-ml311/bin/python"
ARCHIVE="/Volumes/Toshi4TB/Playing Cards.v2-initial.yolov8.zip"
AUDIT="/Volumes/Toshi4TB/DEV_CACHE/CardViper/joshua-artifact-audit-pushed.json"
FAMILY_AUDIT="/Volumes/Toshi4TB/DEV_CACHE/CardViper/joshua-family-audit-pushed.json"
FAMILY_PROPOSAL="/Volumes/Toshi4TB/DEV_CACHE/CardViper/joshua-family-proposal-pushed.json"
REVIEW_DECISIONS="$HOME/Downloads/review-decisions (4).json"
REVIEWED_GROUPS="/Volumes/Toshi4TB/DEV_CACHE/CardViper/reviewed-groups-session-a.json"
# The extraction and crops can be large; keep them off the nearly-full internal SSD.
WORK="/Volumes/Toshi4TB/DEV_CACHE/CardViper/session-a-demo"

test -x "$PY"
test -f "$ARCHIVE"
test -f "$AUDIT"
test -f "$FAMILY_AUDIT"
test -f "$FAMILY_PROPOSAL"
test -f "$REVIEW_DECISIONS"

# Preserve the approved review result off /var/folders before training. If this
# durable file already exists, do not recreate it; it is the approval record.
if [ ! -f "$REVIEWED_GROUPS" ]; then
  PYTHONPATH=ml "$PY" -m cardviper_ml.apply_family_review \
    --archive "$ARCHIVE" \
    --audit "$AUDIT" \
    --family-audit "$FAMILY_AUDIT" \
    --proposal "$FAMILY_PROPOSAL" \
    --review "$REVIEW_DECISIONS" \
    --output "$REVIEWED_GROUPS"
fi
test -f "$REVIEWED_GROUPS"
test ! -e "$WORK"

PYTHONPATH=ml "$PY" -m cardviper_ml.session_a_demo_data \
  --archive "$ARCHIVE" \
  --audit "$AUDIT" \
  --reviewed-groups "$REVIEWED_GROUPS" \
  --output "$WORK/data" \
  --seed 42

PYTHONPATH=ml "$PY" -m cardviper_ml.train_session_a_demo_classifier \
  --data-root "$WORK/data" \
  --output "$WORK/classifier" \
  --seed 42 --epochs 30 --batch-size 32
```

The classifier command asks TensorFlow for the stock MobileNetV3 ImageNet
initialization. If it is not already in the local Keras cache, TensorFlow may
download that public development-time weight file. It never becomes an app
network/model-delivery path: the finished demo uses only the packaged LiteRT
bytes. For an air-gapped build, pre-seed the cache or change the trainer to an
explicit local initialization before starting; do not add cloud model delivery
to the app.

Train and export the detector from the pinned local OD API environment:

```zsh
DETECTOR_PY="$HOME/.venv-detector/bin/python"
ODAPI_ROOT="/absolute/path/to/pinned/models"
BASE_CHECKPOINT="/absolute/path/to/local/fpnlite-320/checkpoint/ckpt-0"

test -x "$DETECTOR_PY"
test -f "$ODAPI_ROOT/research/object_detection/model_main_tf2.py"
test -f "${BASE_CHECKPOINT}.index"

PYTHONPATH=ml "$DETECTOR_PY" -m cardviper_ml.train_session_a_demo_detector \
  --data-root "$WORK/data" \
  --output "$WORK/detector" \
  --odapi-root "$ODAPI_ROOT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --seed 42 --train-steps 10000 --batch-size 8

PYTHONPATH=ml "$DETECTOR_PY" -m cardviper_ml.detector_export_smoke \
  --odapi-root "$ODAPI_ROOT" \
  --output "$WORK/detector-export-smoke"
```

The smoke command must print `INFERENCE_OK`. It only confirms the pinned OD
API export path; it is not an accuracy result.

## Inspect and package before Android sees any bytes

```zsh
ASSETS="$PWD/app/src/sessionADemo/assets/vision"
test ! -e "$ASSETS"

PYTHONPATH=ml "$PY" -m cardviper_ml.export_session_a_demo \
  --classifier-dir "$WORK/classifier" \
  --detector-dir "$WORK/detector" \
  --destination "$WORK/assets"

mkdir -p "$(dirname "$ASSETS")"
cp -R "$WORK/assets" "$ASSETS"

PYTHONPATH=ml "$PY" - <<'PY'
import json
from pathlib import Path

assets = Path("app/src/sessionADemo/assets/vision")
manifest = json.loads((assets / "manifest.json").read_text())
assert manifest["usage"] == "SESSION_A_DEMO_ONLY"
assert manifest["independence"] == "NOT_INDEPENDENT"
assert manifest["face_up_only"] is True
assert manifest["back_supported"] is False
assert set(path.name for path in assets.iterdir()) == {
    "classifier.tflite", "classifier.json", "detector.tflite", "detector.json", "labels.txt", "manifest.json",
}
print("SESSION_A_DEMO_ONLY assets verified")
PY
```

## Assemble and prove variant isolation

```zsh
ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
export ANDROID_HOME

./gradlew \
  test lintDebug lintSessionADemo assembleDebug assembleSessionADemo

DEBUG_APK="$PWD/app/build/outputs/apk/debug/app-debug.apk"
DEMO_APK="$PWD/app/build/outputs/apk/sessionADemo/app-sessionADemo.apk"
APKSIGNER="$ANDROID_HOME/build-tools/36.0.0/apksigner"

test -f "$DEBUG_APK"
test -f "$DEMO_APK"
if unzip -l "$DEBUG_APK" | grep -q 'assets/vision'; then
  echo "ordinary debug APK unexpectedly carries Session A assets" >&2
  exit 1
fi
unzip -l "$DEMO_APK" | grep -q 'assets/vision/classifier.tflite'
"$APKSIGNER" verify --print-certs "$DEMO_APK" | grep 'certificate SHA-256 digest'
```

The ordinary debug APK must contain no Session A model. The demo APK must
contain all six assets. Both remain debug artifacts; neither is promotable.

## Run on the 16 KB emulator and S22+

```zsh
ADB="$ANDROID_HOME/platform-tools/adb"
EMULATOR_SERIAL="$($ADB -e get-serialno)"
PHONE_SERIAL="R5CT113WRJV"

test "$($ADB -s "$EMULATOR_SERIAL" shell getconf PAGE_SIZE | tr -d '\r')" = 16384

ANDROID_HOME="$ANDROID_HOME" ANDROID_SERIAL="$EMULATOR_SERIAL" \
  ./gradlew connectedSessionADemoAndroidTest

$ADB -s "$EMULATOR_SERIAL" install -r "$DEMO_APK"
$ADB -s "$EMULATOR_SERIAL" shell am start -n com.cardviper.app.sessionademo/.MainActivity

NEW_SIGNER="$("$APKSIGNER" verify --print-certs "$DEMO_APK" \
  | awk '/certificate SHA-256 digest/ {print $NF; exit}')"
test -n "$NEW_SIGNER"

# A prior Session A demo, if present, must share this signer before -r updates it.
INSTALLED_DEMO_PATH="$(
  $ADB -s "$PHONE_SERIAL" shell pm path com.cardviper.app.sessionademo 2>/dev/null \
    | tr -d '\r' | sed -n 's/^package://p' | head -n 1 || true
)"
if [ -n "$INSTALLED_DEMO_PATH" ]; then
  TEMP_DIR="$(mktemp -d)"
  $ADB -s "$PHONE_SERIAL" pull "$INSTALLED_DEMO_PATH" "$TEMP_DIR/installed-demo.apk" >/dev/null
  INSTALLED_SIGNER="$("$APKSIGNER" verify --print-certs "$TEMP_DIR/installed-demo.apk" \
    | awk '/certificate SHA-256 digest/ {print $NF; exit}')"
  test "$INSTALLED_SIGNER" = "$NEW_SIGNER" || {
    echo "existing Session A demo signer differs; do not uninstall without preserving its local data" >&2
    exit 1
  }
fi

$ADB -s "$PHONE_SERIAL" install -r "$DEMO_APK"
$ADB -s "$PHONE_SERIAL" shell am start -n com.cardviper.app.sessionademo/.MainActivity
```

If the signer check stops, do not uninstall. Removing the prior demo would
remove its local app data and is a separate decision. The `.sessionademo`
application ID is deliberately separate from the ordinary app, so it does not
replace the ordinary CardViper installation.

On the phone open **IMAGE TEST**, select a full-table Session-A-style still,
and confirm the persistent **SESSION A / FACE-UP DEMO** disclosure, boxes,
labels, yellow `UNSURE` low-confidence crops, and image-local only KO/KISS
preview. Do not test card backs or use results to update a shoe.

## Promotion is prohibited

Success on Session A only proves that the separate on-device LiteRT route can
run. Promotion still requires ordinary B2/B3 evidence: independent grouped
face and BACK data, real detector negatives, held-out evaluation, approved
53-way classifier evidence, detector evidence, and physical acceptance.
