# CardViper V0.2 Android Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Android-side V0.2 still-image recognition pipeline so CardViper can choose a multi-card blackjack image, run deterministic detector/classifier implementations, render labeled boxes and exact card/BACK results, and preview KO/KISS III deltas without touching the active shoe ledger.

**Architecture:** Extend the existing `CardDetector`/`CardRecognizer` boundary into dimension-aware, suspendable pure-Kotlin vision contracts; add a reusable `ImageRecognitionEngine`; then add an Android image loader, dedicated `ImageRecognitionViewModel`, and `IMAGE TEST` Compose route. This Phase A plan deliberately stops at the model boundary: real trained detector/classifier assets and their concrete on-device tensor contracts belong to the separate Phase B model/export plan, so this phase must not guess tensor shapes or ship fabricated neural outputs.

**Tech Stack:** Kotlin 2.4.10, Android/Compose Material 3, Coroutines 1.10.2, Android Photo Picker, existing KO/KISS III engines, JUnit 4, existing GitHub Actions Android pipeline.

**Spec:** `docs/superpowers/specs/2026-09-23-cardviper-v0-2-image-recognition-design.md`

## Global Constraints

- Primary physical target is Google Pixel 7 on Android 16.
- Keep `compileSdk = 37`, `targetSdk = 36`, `minSdk = 26`, Java 17.
- Installed CardViper remains fully offline: no cloud inference, account, analytics, telemetry, or network service.
- Do not add broad storage/media permissions; use Android Photo Picker.
- Exact face identity is 52 rank/suit combinations plus explicit `BACK`.
- A low-confidence face-up result remains visible and contributes to the **V0.2 diagnostic preview only**; it is marked uncertain.
- `BACK` contributes zero to KO and KISS III preview.
- V0.2 image-test results never append, correct, invalidate, or otherwise mutate the active Room ledger/session.
- Low-confidence single-frame observations are not haptic-safe; this phase emits no haptics.
- Use existing `KoStrategy` and `Kiss3Strategy` for preview values; never duplicate their tag tables in UI code.
- Keep the existing stable CI debug signer and `CardViper-debug-apk` artifact behavior unchanged.
- Use TDD and commit after each task. Do not start the next task until the current task's targeted tests pass.
- Do not implement or guess a LiteRT detector/classifier tensor parser until real exported model files and their exact input/output tensor contract exist.

## Phase Split

The approved V0.2 spec contains two independently reviewable systems:

1. **Phase A — this plan:** Android image selection, dimension-aware vision contracts, crop/orchestration logic, confidence/count preview, deterministic fake-model seam, UI, and no-ledger guarantees.
2. **Phase B — separate plan:** dataset/training, detector/classifier export, 53-label contract, standalone offline on-device runtime adapter, Pixel accuracy evaluation, and real model asset integration.

Phase A is useful and testable on its own, but it is **not** permission to claim that real card recognition is finished.

## Review Focus

1. **Very large or oddly shaped photo:** decoding must be bounded and preserve aspect ratio rather than causing an avoidable OOM; Task 3 tests scale calculation and loader failure.
2. **Detector box partly outside the image or degenerate:** crop padding must clamp safely; invalid/zero-area candidates are skipped without crashing the whole image; Task 2 tests all four edges and a degenerate box.
3. **Face-down card:** `BACK` must never be coerced into a fake `PlayingCard` and must contribute zero to both preview systems; Tasks 1 and 2 test this.
4. **Red vs black 2 in KISS III:** exact suit must survive label decoding and drive the existing KISS III strategy correctly; Tasks 1 and 2 test red/black twos.
5. **Active shoe exists while IMAGE TEST runs:** image analysis must not change events, running count, pending review state, or shoe state; Task 4 adds a session-isolation test and Task 5 performs a source-coupling audit.

---

### Task 1: Replace placeholder byte-array vision contracts with exact 53-class domain contracts

**Files:**
- Create: `app/src/main/java/com/cardviper/app/vision/VisionImage.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/CardIdentity.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/CardLabelCodec.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/VisionException.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/UnavailableCardDetector.kt`
- Modify: `app/src/main/java/com/cardviper/app/vision/CardDetector.kt`
- Modify: `app/src/main/java/com/cardviper/app/vision/CardRecognizer.kt`
- Modify: `app/src/main/java/com/cardviper/app/vision/CardRecognition.kt`
- Modify: `app/src/main/java/com/cardviper/app/vision/CardCandidate.kt`
- Modify: `app/src/main/java/com/cardviper/app/vision/NoOpCardDetector.kt`
- Modify: `app/src/main/java/com/cardviper/app/vision/NoOpCardRecognizer.kt`
- Create test: `app/src/test/java/com/cardviper/app/vision/CardLabelCodecTest.kt`
- Create test: `app/src/test/java/com/cardviper/app/vision/VisionImageTest.kt`

**Interfaces:**
- Consumes: existing `PlayingCard`, `CardRank`, `CardSuit`.
- Produces:
  - `data class PixelRect(val left: Int, val top: Int, val right: Int, val bottom: Int)`
  - `data class VisionImage(val width: Int, val height: Int, val rgb: ByteArray)` with `fun crop(rect: PixelRect): VisionImage`
  - `sealed interface CardIdentity { data class Face(val card: PlayingCard) : CardIdentity; data object Back : CardIdentity }`
  - `data class RankedIdentity(val identity: CardIdentity, val confidence: Float)`
  - `data class CardRecognition(val identity: CardIdentity, val confidence: Float, val alternatives: List<RankedIdentity> = emptyList())`
  - `suspend fun CardDetector.detect(image: VisionImage): List<CardCandidate>`
  - `suspend fun CardRecognizer.recognize(crop: VisionImage): CardRecognition`
  - `CardLabelCodec.decode(label: String): CardIdentity`
  - `CardLabelCodec.encode(identity: CardIdentity): String`
  - `VisionModelUnavailableException`

- [ ] **Step 1: Write the 53-label round-trip tests first**

Use canonical ASCII labels suitable for a model label file: `AS`, `2S`, `10H`, `QD`, `KC`, and `BACK`. Generate every face from existing rank/suit enums and assert encode/decode round-trips.

```kotlin
@Test fun all52FacesAndBackRoundTrip() {
    val faces = CardRank.entries.flatMap { rank ->
        CardSuit.entries.map { suit -> CardIdentity.Face(PlayingCard(rank, suit)) }
    }
    assertEquals(52, faces.size)
    (faces + CardIdentity.Back).forEach { identity ->
        assertEquals(identity, CardLabelCodec.decode(CardLabelCodec.encode(identity)))
    }
    assertEquals(CardIdentity.Back, CardLabelCodec.decode("BACK"))
}

@Test fun redAndBlackTwosPreserveSuit() {
    assertEquals(CardSuit.HEARTS,
        (CardLabelCodec.decode("2H") as CardIdentity.Face).card.suit)
    assertEquals(CardSuit.SPADES,
        (CardLabelCodec.decode("2S") as CardIdentity.Face).card.suit)
}
```

Also assert malformed labels throw `IllegalArgumentException`; never silently return a guessed card.

- [ ] **Step 2: Run the label tests and verify RED**

```bash
./gradlew testDebugUnitTest --tests 'com.cardviper.app.vision.CardLabelCodecTest'
```

Expected: FAIL because the identity/codec does not exist yet.

- [ ] **Step 3: Implement `CardIdentity` and `CardLabelCodec` minimally**

Use explicit rank tokens so `TEN` maps to `10`:

```kotlin
private val rankToToken = mapOf(
    CardRank.ACE to "A", CardRank.TWO to "2", CardRank.THREE to "3",
    CardRank.FOUR to "4", CardRank.FIVE to "5", CardRank.SIX to "6",
    CardRank.SEVEN to "7", CardRank.EIGHT to "8", CardRank.NINE to "9",
    CardRank.TEN to "10", CardRank.JACK to "J", CardRank.QUEEN to "Q",
    CardRank.KING to "K",
)
private val suitToToken = mapOf(
    CardSuit.CLUBS to "C", CardSuit.DIAMONDS to "D",
    CardSuit.HEARTS to "H", CardSuit.SPADES to "S",
)
```

`BACK` is not a `PlayingCard` and must stay separate.

- [ ] **Step 4: Write `VisionImage` invariant/crop tests**

Test exact RGB length, full-image crop, center crop, each image edge, and invalid rectangles.

```kotlin
@Test fun cropUsesRgbCoordinatesWithoutRowBleed() {
    val rgb = ByteArray(4 * 3 * 3) { it.toByte() }
    val image = VisionImage(4, 3, rgb)
    val crop = image.crop(PixelRect(left = 1, top = 1, right = 3, bottom = 3))
    assertEquals(2, crop.width)
    assertEquals(2, crop.height)
    assertEquals(12, crop.rgb.size)
}
```

- [ ] **Step 5: Run the image tests and verify RED**

```bash
./gradlew testDebugUnitTest --tests 'com.cardviper.app.vision.VisionImageTest'
```

Expected: FAIL because `VisionImage`/`PixelRect` do not exist.

- [ ] **Step 6: Implement dimension-aware detector/recognizer contracts**

`VisionImage` requires `width > 0`, `height > 0`, and `rgb.size == width * height * 3`. Implement row-by-row crop copying.

Document and validate `CardCandidate.x/y/width/height` as **source-image pixel coordinates**, not normalized coordinates.

```kotlin
interface CardDetector {
    suspend fun detect(image: VisionImage): List<CardCandidate>
}

interface CardRecognizer {
    suspend fun recognize(crop: VisionImage): CardRecognition
}
```

`NoOpCardDetector` remains an empty deterministic fake. `NoOpCardRecognizer` throws `VisionModelUnavailableException("Card classifier is not configured")` because the new recognizer contract is non-null. `UnavailableCardDetector` always throws `VisionModelUnavailableException("Card detector model is not installed")`; Task 4 uses this for honest production wiring before Phase B assets arrive.

- [ ] **Step 7: Run all vision-domain tests**

```bash
./gradlew testDebugUnitTest --tests 'com.cardviper.app.vision.*'
```

Expected: PASS.

- [ ] **Step 8: Run the full unit suite**

```bash
./gradlew test
```

Expected: PASS.

- [ ] **Step 9: Commit and push**

```bash
git add app/src/main/java/com/cardviper/app/vision app/src/test/java/com/cardviper/app/vision
git commit -m "feat: add exact CardViper image vision contracts"
git push
```

Stop here if execution budget is tight. Report the commit SHA before Task 2.

---

### Task 2: Add reusable still-image orchestration and KO/KISS preview logic

**Files:**
- Create: `app/src/main/java/com/cardviper/app/vision/ImageRecognitionEngine.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/DefaultImageRecognitionEngine.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/ImageRecognitionResult.kt`
- Create test: `app/src/test/java/com/cardviper/app/vision/DefaultImageRecognitionEngineTest.kt`

**Interfaces:**
- Consumes: Task 1 `VisionImage`, `PixelRect`, `CardDetector`, `CardRecognizer`, `CardIdentity`; existing `KoStrategy`, `Kiss3Strategy`.
- Produces:

```kotlin
enum class RecognitionStage { DETECTING, CLASSIFYING }

data class RecognitionProgress(
    val stage: RecognitionStage,
    val completed: Int = 0,
    val total: Int = 0,
)

data class ImageCardObservation(
    val candidate: CardCandidate,
    val cropRect: PixelRect,
    val identity: CardIdentity,
    val classifierConfidence: Float,
    val uncertain: Boolean,
    val alternatives: List<RankedIdentity> = emptyList(),
)

data class ImageRecognitionResult(
    val observations: List<ImageCardObservation>,
    val koDelta: Int,
    val kiss3Delta: Int,
    val uncertainCount: Int,
    val skippedCandidates: Int,
)

interface ImageRecognitionEngine {
    suspend fun recognize(
        image: VisionImage,
        onProgress: (RecognitionProgress) -> Unit = {},
    ): ImageRecognitionResult
}
```

Also produce typed `DetectorInferenceException` and `ClassifierInferenceException` in `VisionException.kt`.

- [ ] **Step 1: Write engine tests before implementation**

Create fake detector/recognizer classes local to the test. Cover BACK, uncertainty, exact KISS color behavior, deterministic ordering, no candidates, and typed detector/classifier failures.

```kotlin
@Test fun backIsVisibleButCountsZero() = runTest {
    val result = engineWith(identity = CardIdentity.Back).recognize(image())
    assertEquals(1, result.observations.size)
    assertEquals(CardIdentity.Back, result.observations.single().identity)
    assertEquals(0, result.koDelta)
    assertEquals(0, result.kiss3Delta)
}

@Test fun lowConfidenceFaceStaysInPreviewAndIsMarkedUncertain() = runTest {
    val result = engineWith(
        identity = CardIdentity.Face(PlayingCard(CardRank.FIVE, CardSuit.CLUBS)),
        confidence = 0.55f,
    ).recognize(image())
    assertTrue(result.observations.single().uncertain)
    assertEquals(1, result.koDelta)
    assertEquals(1, result.kiss3Delta)
}

@Test fun kissDifferentiatesRedAndBlackTwo() = runTest {
    val red = resultFor(PlayingCard(CardRank.TWO, CardSuit.HEARTS))
    val black = resultFor(PlayingCard(CardRank.TWO, CardSuit.SPADES))
    assertEquals(0, red.kiss3Delta)
    assertEquals(1, black.kiss3Delta)
}
```

- [ ] **Step 2: Write crop-padding/clamping tests**

Use candidates crossing each edge and one zero/negative-area candidate. For a 100x80 source and 10% padding, every emitted `cropRect` must satisfy:

```kotlin
assertTrue(rect.left >= 0)
assertTrue(rect.top >= 0)
assertTrue(rect.right <= image.width)
assertTrue(rect.bottom <= image.height)
assertTrue(rect.right > rect.left)
assertTrue(rect.bottom > rect.top)
```

A degenerate candidate increments `skippedCandidates` and produces no observation; it must not crash the image.

- [ ] **Step 3: Run the engine tests and verify RED**

```bash
./gradlew testDebugUnitTest --tests 'com.cardviper.app.vision.DefaultImageRecognitionEngineTest'
```

Expected: FAIL because the engine does not exist.

- [ ] **Step 4: Implement engine/result types**

Use an injected default threshold that Phase B can tune from evaluation data:

```kotlin
class DefaultImageRecognitionEngine(
    private val detector: CardDetector,
    private val recognizer: CardRecognizer,
    private val confidentThreshold: Float = 0.80f,
    private val cropPaddingFraction: Float = 0.10f,
) : ImageRecognitionEngine
```

Engine sequence:
1. emit `DETECTING`;
2. call detector and wrap unexpected exceptions as `DetectorInferenceException` while preserving `VisionModelUnavailableException`;
3. sort candidates by top (`y`) then left (`x`), then confidence descending as tie-breaker;
4. pad/clamp/crop from the original `VisionImage`;
5. skip degenerate boxes and increment `skippedCandidates`;
6. emit `CLASSIFYING completed/total` around each valid crop;
7. call recognizer and wrap unexpected exceptions as `ClassifierInferenceException` while preserving `VisionModelUnavailableException`;
8. assemble observations and preview deltas.

- [ ] **Step 5: Reuse existing blackjack strategy code for deltas**

Do not copy tag tables:

```kotlin
private fun previewDelta(
    strategy: CountStrategy,
    observations: List<ImageCardObservation>,
): Int = observations.sumOf { observation ->
    when (val identity = observation.identity) {
        CardIdentity.Back -> 0
        is CardIdentity.Face -> strategy.valueOf(identity.card) ?: 0
    }
}
```

Use `KoStrategy()` and `Kiss3Strategy()`.

- [ ] **Step 6: Run targeted and full tests**

```bash
./gradlew testDebugUnitTest --tests 'com.cardviper.app.vision.*'
./gradlew test
```

Expected: PASS.

- [ ] **Step 7: Commit and push**

```bash
git add app/src/main/java/com/cardviper/app/vision app/src/test/java/com/cardviper/app/vision
git commit -m "feat: add CardViper still-image recognition engine"
git push
```

Report the commit SHA before Task 3.

---

### Task 3: Add bounded Android image decoding and a dedicated image-test ViewModel

**Files:**
- Create: `app/src/main/java/com/cardviper/app/vision/ImageSourceLoader.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/AndroidImageSourceLoader.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/imagetest/ImageRecognitionUiState.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/imagetest/ImageRecognitionViewModel.kt`
- Create test: `app/src/test/java/com/cardviper/app/vision/ImageDecodeSizingTest.kt`
- Create test: `app/src/test/java/com/cardviper/app/ui/imagetest/ImageRecognitionViewModelTest.kt`

**Interfaces:**
- Consumes: Task 2 engine/progress/result/error types.
- Produces:
  - `fun interface ImageSourceLoader { suspend fun load(source: String): VisionImage }`
  - `class AndroidImageSourceLoader(context: Context, maxLongEdge: Int = 4096) : ImageSourceLoader`
  - `enum class ImageTestPhase { IDLE, LOADING_IMAGE, DETECTING, CLASSIFYING, COMPLETE, NO_CARDS, ERROR }`
  - immutable `ImageRecognitionUiState`
  - `fun ImageRecognitionViewModel.analyze(source: String)` and `fun reset()`.

- [ ] **Step 1: Extract and test decode sizing math first**

Keep sizing math pure:

```kotlin
data class DecodeSize(val width: Int, val height: Int)
fun boundedDecodeSize(width: Int, height: Int, maxLongEdge: Int): DecodeSize
```

```kotlin
@Test fun landscape4096CapPreservesAspectRatio() {
    assertEquals(DecodeSize(4096, 2048), boundedDecodeSize(8000, 4000, 4096))
}

@Test fun smallImageIsNotUpscaled() {
    assertEquals(DecodeSize(1200, 800), boundedDecodeSize(1200, 800, 4096))
}
```

Also reject zero/negative dimensions.

- [ ] **Step 2: Run sizing tests RED, then implement**

```bash
./gradlew testDebugUnitTest --tests 'com.cardviper.app.vision.ImageDecodeSizingTest'
```

- [ ] **Step 3: Implement `AndroidImageSourceLoader`**

Requirements:
- parse Photo Picker source with `Uri.parse(source)`;
- use `ContentResolver` and `ImageDecoder` on API 28+;
- use `BitmapFactory` stream decoding with bounds/sample-size on API 26-27;
- cap decoded long edge at 4096 with the tested sizing function;
- convert to RGB `VisionImage` deterministically;
- perform decode/conversion under `Dispatchers.IO`;
- throw `ImageLoadException` for unreadable URI and `ImageDecodeException` for invalid/failed decode.

No broad media/storage permission.

- [ ] **Step 4: Write ViewModel state-machine tests before implementation**

Use fake `ImageSourceLoader` and fake `ImageRecognitionEngine`. Test:
- `IDLE -> LOADING_IMAGE -> DETECTING/CLASSIFYING -> COMPLETE`;
- zero observations produces phase `NO_CARDS` with exact status `NO CARDS DETECTED`;
- image-load error is distinct from image-decode error;
- model-unavailable, detector, and classifier errors map to different user text;
- after any error, a second `analyze()` can reach COMPLETE;
- a new image cancels the previous in-flight analysis and newest source wins.

```kotlin
@Test fun noCardsIsARecoverableNonCrashState() = runTest(dispatcher) {
    model.analyze("content://test/empty")
    advanceUntilIdle()
    assertEquals(ImageTestPhase.NO_CARDS, model.uiState.value.phase)
    assertEquals("NO CARDS DETECTED", model.uiState.value.statusText)
    model.analyze("content://test/cards")
    advanceUntilIdle()
    assertEquals(ImageTestPhase.COMPLETE, model.uiState.value.phase)
}
```

- [ ] **Step 5: Run ViewModel tests and verify RED**

```bash
./gradlew testDebugUnitTest --tests 'com.cardviper.app.ui.imagetest.ImageRecognitionViewModelTest'
```

- [ ] **Step 6: Implement `ImageRecognitionViewModel` separately from shoe state**

The constructor is exactly:

```kotlin
class ImageRecognitionViewModel(
    private val loader: ImageSourceLoader,
    private val engine: ImageRecognitionEngine,
) : ViewModel()
```

It must not accept `SessionManager`, `SessionRepository`, `Room`, or `CardViperViewModel`.

Map failures to stable copy:
- `ImageLoadException` -> `COULD NOT OPEN IMAGE`
- `ImageDecodeException` -> `COULD NOT DECODE IMAGE`
- `VisionModelUnavailableException` -> `VISION MODEL NOT INSTALLED`
- `DetectorInferenceException` -> `CARD DETECTOR FAILED`
- `ClassifierInferenceException` -> `CARD CLASSIFIER FAILED`

Keep the selected decoded `VisionImage` in state so Compose can display exactly the geometry that produced observations.

- [ ] **Step 7: Run targeted and full tests**

```bash
./gradlew testDebugUnitTest --tests 'com.cardviper.app.ui.imagetest.ImageRecognitionViewModelTest'
./gradlew test
```

Expected: PASS.

- [ ] **Step 8: Commit and push**

```bash
git add app/src/main/java/com/cardviper/app/vision app/src/main/java/com/cardviper/app/ui/imagetest app/src/test/java/com/cardviper/app/vision app/src/test/java/com/cardviper/app/ui/imagetest
git commit -m "feat: add CardViper image loading and test state"
git push
```

Report the SHA before Task 4.

---

### Task 4: Add IMAGE TEST navigation, Photo Picker, overlay transform, and results UI

**Files:**
- Modify: `app/src/main/java/com/cardviper/app/ui/navigation/CardViperDestination.kt`
- Modify: `app/src/main/java/com/cardviper/app/ui/home/HomeScreen.kt`
- Modify: `app/src/main/java/com/cardviper/app/ui/CardViperApp.kt`
- Modify: `app/src/main/java/com/cardviper/app/CardViperApplication.kt`
- Modify: `app/src/main/java/com/cardviper/app/MainActivity.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/imagetest/ImageTestScreen.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/imagetest/FitCenterTransform.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/imagetest/VisionImageBitmap.kt`
- Create test: `app/src/test/java/com/cardviper/app/ui/imagetest/FitCenterTransformTest.kt`
- Create test: `app/src/test/java/com/cardviper/app/ui/imagetest/ImageTestSessionIsolationTest.kt`

**Interfaces:**
- Consumes: Task 3 ViewModel/state, existing Compose navigation and Material theme.
- Produces: `CardViperDestination.IMAGE_TEST`, `IMAGE TEST` home button, Photo Picker flow, fitted-image overlay/results screen.

- [ ] **Step 1: Write pure fit-center overlay transform tests**

For image 1000x500 in a 500x500 viewport, expect scale `0.5`, drawn size 500x250, y-offset 125.

```kotlin
@Test fun wideImageLetterboxesVertically() {
    val transform = FitCenterTransform.create(1000f, 500f, 500f, 500f)
    assertEquals(0.5f, transform.scale, 0.0001f)
    assertEquals(0f, transform.offsetX, 0.0001f)
    assertEquals(125f, transform.offsetY, 0.0001f)
}
```

Also test portrait image, equal aspect ratio, and a source box touching bottom/right edges.

- [ ] **Step 2: Run transform tests RED, then implement**

```bash
./gradlew testDebugUnitTest --tests 'com.cardviper.app.ui.imagetest.FitCenterTransformTest'
```

`FitCenterTransform` stays pure; Compose Canvas code consumes it rather than re-implementing geometry.

- [ ] **Step 3: Add navigation and Home entry point**

Add:

```kotlin
IMAGE_TEST("image_test")
```

Extend `HomeScreen` with `onImageTest: () -> Unit` and add an `OutlinedButton` labeled `IMAGE TEST`. Update footer copy to `V0.2 • Offline image test` once the route is present.

- [ ] **Step 4: Build `ImageTestScreen` with system Photo Picker**

Use:

```kotlin
val picker = rememberLauncherForActivityResult(
    contract = ActivityResultContracts.PickVisualMedia(),
) { uri ->
    if (uri != null) onImageSelected(uri.toString())
}
```

Launch with `PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)`.

Screen requirements:
- back action;
- `CHOOSE IMAGE` / `CHOOSE ANOTHER IMAGE`;
- local/offline note;
- selected image generated locally from state's `VisionImage`;
- stage copy `DETECTING CARDS` or `CLASSIFYING n/total`;
- recoverable `NO CARDS DETECTED` and error states;
- complete-state overlay and result panel.

- [ ] **Step 5: Render boxes/labels in the same fit-center coordinate space**

Use the tested transform for image and overlay. Confident face labels use CardViper primary/on-surface colors. Uncertain predictions use a diagnostic yellow. `BACK` is labeled literally `BACK`.

Result panel shows detected count, ordered labels + confidence, uncertainty markers, `KO Δ`, `KISS III Δ`, and uncertain count.

- [ ] **Step 6: Wire the separate image-test ViewModel honestly before models exist**

`CardViperApplication` adds:

```kotlin
val imageSourceLoader: ImageSourceLoader by lazy { AndroidImageSourceLoader(this) }
val imageRecognitionEngine: ImageRecognitionEngine by lazy {
    DefaultImageRecognitionEngine(
        detector = UnavailableCardDetector(),
        recognizer = NoOpCardRecognizer(),
    )
}
```

This means a selected image produces `VISION MODEL NOT INSTALLED` until Phase B supplies real adapters. Do not wire `NoOpCardDetector` in production because that would misleadingly report `NO CARDS DETECTED` when the model is actually absent.

In `MainActivity`, create a second ViewModel factory:

```kotlin
val imageFactory = viewModelFactory {
    initializer {
        ImageRecognitionViewModel(app.imageSourceLoader, app.imageRecognitionEngine)
    }
}
```

Pass both models into `CardViperApp`, add the `IMAGE_TEST` route, and keep `CardViperViewModel` unchanged except for call-site compatibility if needed.

- [ ] **Step 7: Add a session-isolation JVM test**

Create a minimal fake `SessionRepository`, use `SessionManager` to start a fresh KO shoe and add a king, record `eventsBefore` and `snapshotBefore`, run an `ImageRecognitionViewModel` analysis with fake loader/engine, then assert the shoe is byte-for-byte/domain-equal afterward:

```kotlin
assertEquals(eventsBefore, repository.getEvents(sessionId))
assertEquals(snapshotBefore, manager.currentSnapshot())
```

The test must instantiate `ImageRecognitionViewModel(loader, engine)` directly. If the constructor grows a session/repository argument, this test/task fails design review.

- [ ] **Step 8: Run UI/build verification**

```bash
./gradlew test
./gradlew assembleDebugAndroidTest
./gradlew lintDebug
./gradlew assembleDebug
```

Expected: all commands PASS.

- [ ] **Step 9: Commit and push**

```bash
git add app/src/main/java/com/cardviper/app app/src/test/java/com/cardviper/app/ui/imagetest
git commit -m "feat: add CardViper image test UI"
git push
```

Report the SHA and CI run before Task 5.

---

### Task 5: Phase A verification, documentation, and model-handoff contract

**Files:**
- Modify: `README.md`
- Create: `docs/vision/cardviper-v0-2-model-contract.md`
- Leave unchanged unless a genuine failure requires it: `.github/workflows/android-build.yml`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: explicit model integration contract for Phase B and a verified Phase A CI artifact.

- [ ] **Step 1: Write the model handoff document from actual code**

Use this concrete content structure:

```markdown
# CardViper V0.2 Model Contract

## Required assets
- card detector model: multi-card, one semantic CARD class
- card classifier model: 53 outputs matching `CardLabelCodec`
- label file: exactly 53 canonical labels, one per line

## Required detector adapter semantics
Return `CardCandidate` boxes in source-image pixel coordinates plus detector confidence.

## Required classifier adapter semantics
Return `CardRecognition` containing one of the 52 exact faces or `BACK`, confidence, and optional top-N alternatives.

## Frozen label source of truth
`CardLabelCodec` is authoritative. Phase B export tests must compare the packaged label file with the codec's 53-label set.

## Model-specific fields to freeze during Phase B export
Input width/height/channels, input dtype/quantization, normalization, tensor names or indices, detector box format, score/class tensors, NMS ownership, classifier output dtype/quantization, and output-to-label ordering are copied from the exported models and then tested. They are not guessed in Phase A.
```

- [ ] **Step 2: Update README truthfully**

Document that IMAGE TEST plumbing exists but trained model assets are a Phase B deliverable. Do not say automatic card recognition works while `UnavailableCardDetector` is production wiring.

Add the pipeline:

```text
Photo Picker -> detector -> source crop -> 53-way classifier -> overlay + KO/KISS preview
```

- [ ] **Step 3: Run final verification fresh**

```bash
./gradlew test
./gradlew assembleDebugAndroidTest
./gradlew lintDebug
./gradlew assembleDebug
```

Do not reuse earlier task evidence for the final completion claim.

- [ ] **Step 4: Audit forbidden coupling and permissions**

```bash
git grep -nE 'appendEvent|manualAdd|correctCard|invalidateCard' -- app/src/main/java/com/cardviper/app/ui/imagetest app/src/main/java/com/cardviper/app/vision
```

Expected: no image-test/vision production code invokes shoe-ledger mutation APIs.

```bash
git grep -nE 'INTERNET|READ_MEDIA_IMAGES|READ_EXTERNAL_STORAGE|WRITE_EXTERNAL_STORAGE' -- app/src/main/AndroidManifest.xml
```

Expected: no matches.

Also run:

```bash
git diff --check HEAD~1..HEAD
```

Expected: no whitespace errors in the most recent checkpoint. If Task 5 modifies multiple commits, expand the range to the Phase A base commit.

- [ ] **Step 5: Commit and push the Phase A checkpoint**

```bash
git add README.md docs/vision/cardviper-v0-2-model-contract.md
git commit -m "docs: define CardViper V0.2 model handoff"
git push
```

- [ ] **Step 6: Verify GitHub Actions rather than assuming**

Confirm the pushed workflow finishes successfully, including unit tests, instrumented-test compile, lint, APK build, stable signer verification, and artifact upload. Record commit SHA, workflow run ID, `CardViper-debug-apk` artifact ID, and any warning requiring follow-up.

## Phase A Acceptance

Phase A is accepted only when:
- all automated verification is green;
- IMAGE TEST can select/decode a photo without broad storage permission;
- fake detector/classifier tests prove multi-card boxes, exact face identities, BACK, uncertainty, and KO/KISS preview behavior;
- image-test code cannot mutate the shoe ledger;
- the UI remains recoverable after no-card/error states;
- production wiring says `VISION MODEL NOT INSTALLED` until real assets exist;
- README clearly states trained neural assets are still Phase B.

**Do not call V0.2 finished at this point.** The next plan is Phase B: acquire/train/export the detector and 53-way classifier, freeze their concrete tensor contract, add the standalone offline on-device runtime adapter, package the assets, and run the Pixel 7 physical acceptance set from the approved spec.
