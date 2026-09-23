# CardViper V0.1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the first installable CardViper Android APK with a real Pixel 7 camera preview, KO/KISS III domain engines, append-only ledger, Room persistence/recovery, manual correction plumbing, a sleek Compose HUD, and GitHub Actions artifact delivery.

**Architecture:** The app separates camera/vision, physical tracking, event persistence, count strategies, and UI. Room persists immutable card/session facts; a resolver builds the effective card list; KO and KISS III replay that list into derived snapshots. Final neural vision is behind interfaces so V0.1 remains stable and testable without a trained model.

**Tech Stack:** Kotlin, Gradle Kotlin DSL, Jetpack Compose, Material 3, CameraX, Room, DataStore, coroutines/Flow, JUnit, AndroidX testing, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-22-cardviper-foundation-design.md`

## Global Constraints
- App name: CardViper.
- Package: `com.cardviper.app`.
- Primary target: Google Pixel 7 on Android 16.
- Minimum SDK: 26 unless a selected current stable dependency requires a documented higher floor.
- Portrait-first dark UI with restrained green accents.
- Offline only: no networking, auth, analytics, ads, or cloud services.
- KO is the default count; KISS III is alternate.
- Default shoe is 6 decks.
- KO V1 tags: 2-7 +1, 8-9 0, 10-A -1; IRC formula `4 - (4 * decks)`.
- KISS III V1 tags: black 2 +1, red 2 0, 3-7 +1, 8-9 0, 10-A -1; six-deck IRC 9, key 20, insurance 25.
- Persist facts; derive count state.
- One physical card can create at most one countable commit event per live track lifecycle.
- Low-confidence/unresolved cards never change the count.
- No destructive Room migration in the production configuration.
- Final detector/recognizer model/runtime is out of V0.1 scope.

## Review Focus
1. Process death during an active shoe must restore the same effective ledger/count state; covered in Task 5 recovery tests.
2. Repeated frames/reacquisition of one card must not duplicate the ledger event; covered in Task 4 tracker tests.
3. KISS III must not score a 2 until red/black is known; covered in Task 2 strategy tests and Task 4 eligibility tests.
4. Corrections/invalidation/undo must replay deterministically without arithmetic drift; covered in Task 3 resolver tests.
5. Camera permission denial/retry and app background/resume must leave the shoe intact; covered in Task 6 UI/device behavior and Task 9 acceptance checks.

---

### Task 1: Bootstrap the Android project and build shell

**Files:**
- Create: `settings.gradle.kts`
- Create: `build.gradle.kts`
- Create: `gradle.properties`
- Create: `gradle/libs.versions.toml`
- Create: `app/build.gradle.kts`
- Create: `app/src/main/AndroidManifest.xml`
- Create: `app/src/main/java/com/cardviper/app/CardViperApplication.kt`
- Create: `app/src/main/java/com/cardviper/app/MainActivity.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/theme/CardViperTheme.kt`
- Create: `app/src/main/res/values/strings.xml`
- Create: `app/src/main/res/values/themes.xml`
- Create: `.gitignore`

**Interfaces:**
- Produces: a standard single-module Android application with Compose enabled and dependencies for CameraX, Room, DataStore, coroutines, navigation, and tests.

- [ ] **Step 1: Scaffold the Gradle project using mutually compatible current stable Android/Kotlin libraries.**

Use a version catalog. Configure `namespace = "com.cardviper.app"`, `applicationId = "com.cardviper.app"`, minSdk 26 (or the lowest documented compatible floor if a selected current stable library requires higher), and compile/target SDK compatible with Android 16 tooling.

- [ ] **Step 2: Create the smallest Compose launch surface.**

`MainActivity` should render a dark CardViper theme and a centered `CARDVIPER` title plus `Foundation boot OK`. Do not add camera or domain logic yet.

- [ ] **Step 3: Run a clean build.**

Run:
```bash
./gradlew clean assembleDebug
```
Expected: BUILD SUCCESSFUL and an APK under `app/build/outputs/apk/debug/`.

- [ ] **Step 4: Run unit tests and lint even though test volume is initially small.**

Run:
```bash
./gradlew test lintDebug
```
Expected: BUILD SUCCESSFUL.

- [ ] **Step 5: Commit.**

```bash
git add .
git commit -m "feat: bootstrap CardViper Android foundation"
```

---

### Task 2: Implement pure card and count-strategy domain logic with tests

**Files:**
- Create: `app/src/main/java/com/cardviper/app/model/CardRank.kt`
- Create: `app/src/main/java/com/cardviper/app/model/CardSuit.kt`
- Create: `app/src/main/java/com/cardviper/app/model/CardColor.kt`
- Create: `app/src/main/java/com/cardviper/app/model/PlayingCard.kt`
- Create: `app/src/main/java/com/cardviper/app/blackjack/CountStrategyId.kt`
- Create: `app/src/main/java/com/cardviper/app/blackjack/CountSnapshot.kt`
- Create: `app/src/main/java/com/cardviper/app/blackjack/CountStrategy.kt`
- Create: `app/src/main/java/com/cardviper/app/blackjack/KoStrategy.kt`
- Create: `app/src/main/java/com/cardviper/app/blackjack/Kiss3Strategy.kt`
- Test: `app/src/test/java/com/cardviper/app/blackjack/KoStrategyTest.kt`
- Test: `app/src/test/java/com/cardviper/app/blackjack/Kiss3StrategyTest.kt`

**Interfaces:**
- Produces: `CountStrategy.valueOf(card: PlayingCard): Int?`, `initialRunningCount(deckCount: Int): Int`, and `evaluate(startingCount: Int, cards: List<PlayingCard>): CountSnapshot`.
- `Int?` is intentional: KISS III returns null for a rank-2 card whose color cannot yet be established.

- [ ] **Step 1: Write failing KO tests.**

Tests must assert every rank tag, `initialRunningCount(1)==0`, `(2)==-4`, `(4)==-12`, `(6)==-20`, `(8)==-28`, and a sequence `2,5,K,8,A,7` has net delta +1.

- [ ] **Step 2: Run KO tests and confirm failure.**

```bash
./gradlew test --tests '*KoStrategyTest'
```
Expected: FAIL because production strategy classes do not yet exist.

- [ ] **Step 3: Implement minimal KO domain code and rerun.**

Expected: all KO tests PASS.

- [ ] **Step 4: Write failing KISS III tests.**

Assert:
- 2 spades/clubs = +1
- 2 hearts/diamonds = 0
- unresolved 2 color = null
- 3-7 = +1
- 8-9 = 0
- 10/J/Q/K/A = -1
- six-deck `initialRunningCount(6)==9`
- six-deck profile exposes key 20 and insurance 25

- [ ] **Step 5: Implement minimal KISS III logic and rerun.**

```bash
./gradlew test --tests '*Kiss3StrategyTest'
```
Expected: PASS.

- [ ] **Step 6: Add full-deck invariant tests.**

Generate all 52 distinct cards in test code and assert KO deck net = +4. Assert KISS III V1 deck net = +2 when exactly two black twos score +1, all 3-7 score +1, and 10-A score -1. Verify six repeated decks produce six times each per-deck delta from the appropriate IRC.

- [ ] **Step 7: Run all domain tests.**

```bash
./gradlew test
```
Expected: PASS.

- [ ] **Step 8: Commit.**

```bash
git add app/src/main/java/com/cardviper/app/model app/src/main/java/com/cardviper/app/blackjack app/src/test/java/com/cardviper/app/blackjack
git commit -m "feat: add KO and KISS III count engines"
```

---

### Task 3: Implement append-only ledger and deterministic resolver

**Files:**
- Create: `app/src/main/java/com/cardviper/app/model/LedgerEventType.kt`
- Create: `app/src/main/java/com/cardviper/app/model/CardEventSource.kt`
- Create: `app/src/main/java/com/cardviper/app/model/CardLedgerEvent.kt`
- Create: `app/src/main/java/com/cardviper/app/blackjack/ResolvedCard.kt`
- Create: `app/src/main/java/com/cardviper/app/blackjack/LedgerResolver.kt`
- Create: `app/src/main/java/com/cardviper/app/blackjack/SessionCalculator.kt`
- Test: `app/src/test/java/com/cardviper/app/blackjack/LedgerResolverTest.kt`
- Test: `app/src/test/java/com/cardviper/app/blackjack/SessionCalculatorTest.kt`

**Interfaces:**
- Consumes: `PlayingCard`, `CountStrategy` from Task 2.
- Produces: `LedgerResolver.resolve(events): List<ResolvedCard>` and `SessionCalculator.calculate(startingCount, strategy, resolvedCards): CountSnapshot`.

- [ ] **Step 1: Write failing resolver tests for commit, correction, invalidation, manual add, double correction, and invalidating a corrected card.**

Use this core fixture:
```text
001 COMMIT K♥
002 COMMIT 6♣
003 CORRECT 001 -> Q♥
004 INVALIDATE 002
005 MANUAL_ADD 3♦
```
Expected effective sequence: `Q♥, 3♦`.

- [ ] **Step 2: Run resolver tests and confirm failure.**

```bash
./gradlew test --tests '*LedgerResolverTest'
```

- [ ] **Step 3: Implement resolver with immutable event semantics.**

Invalid target references must be ignored/rejected deterministically with a typed resolver result or exception chosen consistently; never mutate earlier events.

- [ ] **Step 4: Write and implement replay tests.**

Test a ledger under KO, correct a high card to a low card, and assert the count equals a full fresh replay rather than an incremental guessed adjustment. Then switch KO -> KISS III -> KO and assert raw ledger/event IDs remain unchanged and the second KO snapshot equals the first.

- [ ] **Step 5: Run all tests.**

```bash
./gradlew test
```
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add app/src/main/java/com/cardviper/app app/src/test/java/com/cardviper/app
git commit -m "feat: add append-only blackjack ledger resolver"
```

---

### Task 4: Add live tracking domain, pending-review rules, and exactly-once protection

**Files:**
- Create: `app/src/main/java/com/cardviper/app/tracking/TrackLifecycle.kt`
- Create: `app/src/main/java/com/cardviper/app/tracking/FaceState.kt`
- Create: `app/src/main/java/com/cardviper/app/tracking/TableZone.kt`
- Create: `app/src/main/java/com/cardviper/app/tracking/TrackedCard.kt`
- Create: `app/src/main/java/com/cardviper/app/tracking/TrackDecision.kt`
- Create: `app/src/main/java/com/cardviper/app/tracking/CardCommitGate.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/CardCandidate.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/CardRecognition.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/CardDetector.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/CardRecognizer.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/NoOpCardDetector.kt`
- Create: `app/src/main/java/com/cardviper/app/vision/NoOpCardRecognizer.kt`
- Test: `app/src/test/java/com/cardviper/app/tracking/CardCommitGateTest.kt`

**Interfaces:**
- Produces: a pure `CardCommitGate` that decides `Commit(card)`, `Pending(reason)`, or `NoAction` from current track state and active strategy requirements.

- [ ] **Step 1: Write failing exactly-once tests.**

Test 100 observations of one confirmed face-up track and assert only its first eligible transition can emit `Commit`; later observations emit `NoAction`.

- [ ] **Step 2: Add occlusion/reacquisition and new-identical-card tests.**

A counted track that becomes temporarily missing and returns with the same track ID must not recommit. A genuinely new track ID representing the same rank/suit may commit independently.

- [ ] **Step 3: Add face-down/reveal and KISS III unresolved-two tests.**

Face-down is never eligible. When the same track becomes face-up and confidently identified, it may commit once. In KISS III a `2` with unknown color must return Pending; when black/red resolves it may commit with the appropriate strategy value.

- [ ] **Step 4: Implement the minimal tracking/vision interfaces and gate.**

Do not build a neural model. No-op detector/recognizer return empty/no recognition so the app can run safely.

- [ ] **Step 5: Run tests.**

```bash
./gradlew test --tests '*CardCommitGateTest'
./gradlew test
```
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add app/src/main/java/com/cardviper/app/tracking app/src/main/java/com/cardviper/app/vision app/src/test/java/com/cardviper/app/tracking
git commit -m "feat: add CardViper tracking and vision boundaries"
```

---

### Task 5: Add Room persistence, repositories, DataStore, and crash recovery

**Files:**
- Create: `app/src/main/java/com/cardviper/app/data/db/CardViperDatabase.kt`
- Create: `app/src/main/java/com/cardviper/app/data/db/ShoeSessionEntity.kt`
- Create: `app/src/main/java/com/cardviper/app/data/db/CardLedgerEventEntity.kt`
- Create: `app/src/main/java/com/cardviper/app/data/db/PendingReviewEntity.kt`
- Create: `app/src/main/java/com/cardviper/app/data/db/HardCaseSampleEntity.kt`
- Create: `app/src/main/java/com/cardviper/app/data/db/CardViperDao.kt`
- Create: `app/src/main/java/com/cardviper/app/data/SessionRepository.kt`
- Create: `app/src/main/java/com/cardviper/app/data/RoomSessionRepository.kt`
- Create: `app/src/main/java/com/cardviper/app/data/CardViperPreferences.kt`
- Create: `app/src/main/java/com/cardviper/app/data/PreferencesRepository.kt`
- Create: `app/src/main/java/com/cardviper/app/session/SessionState.kt`
- Create: `app/src/main/java/com/cardviper/app/session/StartMode.kt`
- Create: `app/src/main/java/com/cardviper/app/session/ShoeSession.kt`
- Create: `app/src/main/java/com/cardviper/app/session/SessionSnapshot.kt`
- Create: `app/src/main/java/com/cardviper/app/session/SessionManager.kt`
- Test: `app/src/androidTest/java/com/cardviper/app/data/CardViperDatabaseTest.kt`
- Test: `app/src/test/java/com/cardviper/app/session/SessionManagerTest.kt`

**Interfaces:**
- Consumes: ledger resolver and strategies.
- Produces: repository flows for active session, ledger, pending reviews, and derived `SessionSnapshot`.

- [ ] **Step 1: Define Room schema version 1 and DAO operations.**

Use transactions for commit+pending-resolution and correction/invalidation operations. Do not persist running count as canonical truth.

- [ ] **Step 2: Write session-manager tests using an in-memory/fake repository.**

Test fresh 6D KO starts at -20; six-deck KISS III starts at 9; switching strategy replays the same ledger; pause/end semantics; manual add/invalidate; penetration uses effective card count.

- [ ] **Step 3: Implement SessionManager until JVM tests pass.**

```bash
./gradlew test --tests '*SessionManagerTest'
```
Expected: PASS.

- [ ] **Step 4: Write Room instrumented transaction/recovery tests.**

Create a session with commits, correction, manual add, and invalidation; close/reopen the Room DB; reload facts; resolve/recalculate; assert the same derived values. Force a transaction failure and assert partial state is not committed.

- [ ] **Step 5: Implement DataStore defaults.**

Defaults: KO, 6 decks, auto attention on, correction crops on, uncertain crops on, confidence off, recent cards on, vision debug off.

- [ ] **Step 6: Run JVM tests and compile instrumented tests.**

```bash
./gradlew test assembleDebug assembleDebugAndroidTest
```
Expected: PASS/BUILD SUCCESSFUL. Run connected tests when an emulator/device is available; do not fake results if no device exists.

- [ ] **Step 7: Commit.**

```bash
git add app/src/main/java/com/cardviper/app/data app/src/main/java/com/cardviper/app/session app/src/test app/src/androidTest
git commit -m "feat: persist CardViper shoes and ledger state"
```

---

### Task 6: Build Compose navigation, live HUD, CameraX preview, and permission flow

**Files:**
- Modify: `app/src/main/AndroidManifest.xml`
- Modify: `app/src/main/java/com/cardviper/app/MainActivity.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/CardViperApp.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/navigation/CardViperDestination.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/home/HomeScreen.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/shoe/NewShoeScreen.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/live/LiveViperScreen.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/live/CameraPreview.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/live/LiveHud.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/live/RecentCardsStrip.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/review/ReviewScreen.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/settings/SettingsScreen.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/CardViperViewModel.kt`
- Test: `app/src/test/java/com/cardviper/app/ui/CardViperViewModelTest.kt`

**Interfaces:**
- Consumes: SessionManager flows/actions.
- Produces: Home -> New Shoe -> Live Vision flow, Resume active shoe, mode switching, manual add, undo/invalidate, settings/review shells.

- [ ] **Step 1: Add camera permission and CameraX preview.**

Request `android.permission.CAMERA` only when entering/starting live vision. Permission denial must show a clear retry action and must not destroy the session.

- [ ] **Step 2: Implement Home and New Shoe screens.**

Home shows Resume Shoe when active. New Shoe defaults to KO + 6 decks + Fresh and supports KO/KISS III plus 1/2/4/6/8 decks.

- [ ] **Step 3: Implement Live HUD.**

Camera fills most of portrait screen. Bottom HUD shows mode, large RC, strategy-specific references, penetration/deck estimate, recent cards, Review count, `+ CARD`, `UNDO`, and overflow shell. With no real recognizer, overlay should remain stable and simply show zero tracked cards.

- [ ] **Step 4: Implement manual add and undo.**

`+ CARD` opens a compact rank selector and, when KISS III rank 2 is chosen, requires red/black or suit before commit. `UNDO` appends an invalidation event targeting the most recent effective card; it never deletes ledger history.

- [ ] **Step 5: Implement mode switch mid-shoe.**

Switching KO/KISS III recalculates from the same ledger and preserves event IDs/history.

- [ ] **Step 6: Add ViewModel tests.**

Assert permission-independent state flows, new shoe defaults, strategy switch, manual add, undo, and app/background state do not reset an active session.

- [ ] **Step 7: Run tests/lint/build.**

```bash
./gradlew test lintDebug assembleDebug
```
Expected: BUILD SUCCESSFUL.

- [ ] **Step 8: Commit.**

```bash
git add app/src/main
git add app/src/test
git commit -m "feat: add CardViper HUD and CameraX workflow"
```

---

### Task 7: Complete correction/review and hard-case plumbing

**Files:**
- Create: `app/src/main/java/com/cardviper/app/ui/review/CardCorrectionSheet.kt`
- Create: `app/src/main/java/com/cardviper/app/ui/review/PendingReviewSheet.kt`
- Modify: `app/src/main/java/com/cardviper/app/ui/live/RecentCardsStrip.kt`
- Modify: `app/src/main/java/com/cardviper/app/data/SessionRepository.kt`
- Modify: `app/src/main/java/com/cardviper/app/data/RoomSessionRepository.kt`
- Test: `app/src/test/java/com/cardviper/app/session/CorrectionFlowTest.kt`

**Interfaces:**
- Produces: correct identity, invalidate false card, resolve pending candidate, and hard-case metadata capture.

- [ ] **Step 1: Write correction-flow tests.**

A corrected committed card must append CARD_CORRECTED and recalculate. A false positive must append CARD_INVALIDATED. A pending card must not affect count until manual/auto resolution. Hard-case metadata should be created for corrected/false-positive/low-confidence cases when configured.

- [ ] **Step 2: Implement correction sheet.**

Tap a recent card -> compact rank selector; suit/color only when needed. Commit correction in one repository transaction where applicable.

- [ ] **Step 3: Implement pending review sheet.**

Show best candidate/alternates when present and allow manual resolve/discard. No fake recognition alternatives are required in V0.1; the UI must handle an empty/no-op vision source gracefully.

- [ ] **Step 4: Run tests and build.**

```bash
./gradlew test lintDebug assembleDebug
```
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add app/src/main app/src/test
git commit -m "feat: add CardViper correction and review flow"
```

---

### Task 8: Add CI APK artifact pipeline and project documentation

**Files:**
- Create: `.github/workflows/android-build.yml`
- Modify: `README.md`

**Interfaces:**
- Produces: repeatable cloud build on push, pull request, and manual dispatch; downloadable `CardViper-debug-apk` artifact.

- [ ] **Step 1: Create GitHub Actions workflow.**

Workflow triggers:
```yaml
on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:
```

Steps must checkout, install the JDK required by the selected AGP, enable Gradle caching, run:
```bash
./gradlew test
./gradlew lintDebug
./gradlew assembleDebug
```
and upload `app/build/outputs/apk/debug/app-debug.apk` as artifact name `CardViper-debug-apk`.

- [ ] **Step 2: Replace README with real usage/build instructions.**

README must include:
- `# CardViper`
- description: blackjack computer-vision training platform
- default KO + alternate KISS III
- current V0.1 scope
- architecture: Camera -> Vision -> Tracking -> Ledger -> Blackjack Brain -> HUD
- phone download path: GitHub -> Actions -> Android Build -> latest successful run -> Artifacts -> CardViper-debug-apk
- Pixel install note for allowing APK installs from the browser/file source used
- roadmap: V0.1 foundation, V0.2 image test, V0.3 video-file recognition, V0.4 live recognition, then tracker/ROI/model refinement

- [ ] **Step 3: Run the exact CI commands locally in the Codex environment.**

```bash
./gradlew test
./gradlew lintDebug
./gradlew assembleDebug
```
Expected: all PASS. If the environment lacks an Android SDK component, install/configure only what the build actually requires and document it; do not claim success without execution.

- [ ] **Step 4: Commit.**

```bash
git add .github/workflows/android-build.yml README.md
git commit -m "ci: build and publish CardViper debug APK"
```

---

### Task 9: Final verification and release-ready V0.1 handoff

**Files:**
- Modify only files required to fix discovered verification failures.

**Interfaces:**
- Produces: a clean main-branch candidate whose automated build emits an installable debug APK.

- [ ] **Step 1: Fresh-clone verification.**

From a clean checkout with no untracked build files, run:
```bash
./gradlew test
./gradlew lintDebug
./gradlew assembleDebug
```
Expected: all succeed.

- [ ] **Step 2: Verify APK existence.**

```bash
test -f app/build/outputs/apk/debug/app-debug.apk
```
Expected: exit status 0.

- [ ] **Step 3: Verify no prohibited dependencies/features.**

Search project for networking SDKs, analytics, ads, auth, secrets, absolute developer paths, and destructive Room migration. Remove any accidental additions.

- [ ] **Step 4: Verify core automated behaviors.**

Confirm test output includes KO, KISS III, ledger replay, correction/invalidation, strategy switching, exactly-once track commit, pending review, and recovery tests.

- [ ] **Step 5: Device acceptance checklist for Pixel 7.**

When the user installs the APK, expected behavior:
1. app launches without crash
2. New Shoe defaults to KO / 6 decks / Fresh
3. Start Viper requests camera permission
4. granting permission shows stable live rear-camera preview
5. manual `+ CARD` changes RC correctly
6. Undo reverses the effective card without deleting history
7. switch to KISS III recalculates the same ledger
8. KISS III manual black/red 2 behaves correctly
9. background/foreground preserves active shoe
10. force-close/reopen offers/resumes the same active session state

- [ ] **Step 6: Fix every discovered build/test defect, rerunning the narrow failing test first and then the complete suite.**

- [ ] **Step 7: Commit final verification fixes if any.**

```bash
git add .
git commit -m "fix: finalize CardViper V0.1 verification"
```
Skip this commit only when there are genuinely no changes.

## Implementation completion report
At the end, report only verified facts:
- final commit SHA
- `test` result
- `lintDebug` result
- `assembleDebug` result
- exact APK path
- whether GitHub Actions workflow is present
- any device-only checks that still require the Pixel 7
- no claim that neural card recognition exists in V0.1
