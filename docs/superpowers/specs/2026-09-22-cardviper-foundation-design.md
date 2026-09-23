# CardViper Foundation Design

## Goal
Build CardViper as an offline-first Android blackjack computer-vision training platform for a Google Pixel 7, with KO as the default count and KISS III as an alternate mode. V0.1 must be a stable, installable Android foundation with a real camera preview, persistent shoe/session state, testable count engines, correction/review plumbing, and CI-built APKs. Final card recognition is deliberately deferred behind interfaces.

## Primary device and viewing geometry
- Primary development/test device: Google Pixel 7, Android 16.
- Secondary phone may be used for control/reference only.
- Primary eventual camera geometry: chest-level/first-person view.
- Must also tolerate wider whole-table views.
- Camera architecture therefore uses a full-frame awareness pass plus dynamic high-resolution regions of interest rather than a single fixed crop.

## Product principles
- Tool first, trainer second.
- Offline only: no networking, accounts, analytics, ads, or cloud services.
- Persist facts; derive count state.
- A physical card may affect the running count exactly once.
- Uncertain recognition never changes the count until resolved.
- Computer vision reports cards; count strategies interpret cards.
- The same card ledger must be replayable through KO or KISS III without retracking.
- Corrections are auditable events, not destructive history edits.

## Android foundation
- App name: CardViper.
- Package: `com.cardviper.app`.
- Kotlin.
- Gradle Kotlin DSL.
- Jetpack Compose + Material 3.
- Portrait-first.
- Dark black/charcoal UI with restrained green accents.
- CameraX for V0.1 live preview.
- Room for session/event persistence.
- DataStore for preferences.
- Minimum SDK 26 unless dependency constraints require a higher documented minimum.
- Current stable compile/target SDK supported by the selected Android Gradle Plugin.

## Domain model

### PlayingCard
- rank: A, 2-10, J, Q, K
- suit: clubs, diamonds, hearts, spades when known
- color is derived from suit when suit is known
- recognition may temporarily know rank plus red/black without knowing exact suit

### CountStrategy
Pure domain interface. No Android, Room, CameraX, or UI dependencies.

Responsibilities:
- strategy identifier and version
- initial running count for deck count
- per-card tag value
- strategy reference values needed by the HUD
- replay/evaluation of an effective ledger into a CountSnapshot

### KO V1
Tags:
- 2-7: +1
- 8-9: 0
- 10/J/Q/K/A: -1

Use classic KO-style IRC formula `4 - (4 * decks)` for supported whole-deck configurations, giving 6 decks = -20. Keep strategy constants versioned.

### KISS III V1
Use the Fred Renzey-style color-aware form:
- black 2: +1
- red 2: 0
- 3-7: +1
- 8-9: 0
- 10/J/Q/K/A: -1

For the default six-deck profile expose:
- IRC: 9
- key count: 20
- insurance count: 25

Store KISS III profile/reference values behind the strategy implementation so they can be revised or extended without contaminating camera/tracking/session code.

## Physical-card lifecycle
Core lifecycle:
`NEW -> CONFIRMED -> COUNTED -> EXITED`

TrackedCard live state may also carry:
- trackId
- bounding box
- center and motion estimate
- firstSeen/lastSeen
- stableFrames/missingFrames
- rank/suit/color candidates and confidence
- face state: FACE_UP, FACE_DOWN, UNKNOWN
- zone: PLAYER, DEALER, TABLE, DISCARD, UNKNOWN

Rules:
- NEW never changes count.
- CONFIRMED means the physical object is believed to be a card with stable identity evidence.
- CONFIRMED may emit exactly one reveal/commit event when eligible.
- COUNTED cannot emit a second count event simply because it remains visible.
- Temporary occlusion should preserve/reacquire the same track.
- EXITED tracks may remain in a short graveyard cache to avoid duplicate resurrection.
- Two identical ranks/suits in a multi-deck shoe are still two physical cards if spatial/temporal evidence indicates two tracks.
- Face-down cards remain uncounted until revealed.

## Vision architecture boundary
V0.1 defines interfaces and fake implementations; final model/runtime is not selected yet.

Pipeline target:
`Camera -> wide detector -> high-res crops -> perspective rectification -> rank/color/suit recognition -> temporal consensus -> physical tracker -> ledger -> count strategy -> HUD`

Target behavior for later versions:
- Camera 30 fps or device-appropriate smooth preview.
- Wide awareness detector roughly 6-15 Hz depending on activity.
- Use lower-resolution awareness frames, then crop candidates from the original frame.
- Dynamic ROI favors lower/central player-card region but periodically scans whole frame.
- Recognition votes over multiple frames rather than trusting one frame.
- A committed high-confidence identity has hysteresis and is not rewritten by one blurry contradictory frame.
- SEARCH, ACTIVE, and LOCKED workload states should eventually reduce heat/battery use.

Interfaces to establish in V0.1:
- `CardDetector`
- `CardRecognizer`
- fake/no-op implementations that allow the rest of the app to run without a neural model

## Ledger and corrections
The permanent source of truth is an append-only event ledger.

Event types:
- CARD_COMMITTED
- CARD_CORRECTED
- CARD_INVALIDATED
- CARD_MANUAL_ADDED

Each event includes:
- eventId
- sessionId
- sequence number
- timestamp
- target event id when applicable
- track id when applicable
- rank/suit/color when applicable
- source: VISION, MANUAL, CORRECTION
- rank/color/suit confidence when applicable
- crop reference when applicable

A LedgerResolver converts raw events into the effective card sequence. Count engines consume only the effective sequence.

Manual correction requirements:
- recent-card strip is tappable
- wrong identity can be corrected
- false card can be invalidated
- missed card can be manually added
- Undo Last invalidates/reverses the most recent effective card operation without deleting history
- any correction triggers full deterministic ledger replay

## Pending review
Low-confidence candidates do not affect count.

PendingReview states:
- PENDING
- AUTO_RESOLVED
- MANUALLY_RESOLVED
- DISCARDED

A pending candidate can later auto-resolve from stronger evidence or be manually resolved. Resolution creates the normal ledger event.

## Hard-case collection
Persist metadata for:
- WRONG_RANK
- WRONG_SUIT
- WRONG_COLOR
- FALSE_POSITIVE
- LOW_CONFIDENCE
- MANUAL_MISSED_CARD
- TRACKING_FAILURE

Do not store image bytes in Room. Room stores a crop reference; files live in app storage. By default retain corrected, uncertain, false-positive, and explicitly selected hard-case crops rather than every normal card.

## Shoe session
Session states:
- READY
- ACTIVE
- PAUSED
- ENDED

Session data:
- sessionId
- strategy id and version
- nominal decks
- start mode: FRESH or MID_SHOE
- starting running count
- optional starting deck estimate
- timestamps

Rules:
- fresh shoe starts from strategy IRC
- manual New Shoe is authoritative in early versions; no automatic shuffle reset in V0.1
- Pause preserves shoe state while stopping analysis
- End Shoe archives the session
- switching KO/KISS III mid-shoe replays the same ledger; it never resets or mutates card history
- crash recovery reloads active session + ledger + pending reviews and recalculates derived state
- live spatial tracks are not persisted across process death

Derived values are not canonical DB truth:
- running count
- cards seen
- penetration
- estimated decks remaining
- peak count
- correction/manual/false-positive totals

For a fresh shoe, simple penetration is `effectiveCardsSeen / (nominalDecks * 52)` and estimated decks remaining may begin as `(nominalDecks * 52 - effectiveCardsSeen) / 52`. These are contextual estimates, not true-count requirements.

## Persistence
Use Room entities for:
- ShoeSession
- CardLedgerEvent
- PendingReview
- HardCaseSample

Use DataStore for preferences:
- default count mode = KO
- default decks = 6
- auto attention = on
- save correction crops = on
- save uncertain crops = on
- show confidence = off
- show recent cards = on
- vision debug = off

Important DB rules:
- meaningful commits/corrections use Room transactions
- schema is versioned from day one
- do not use destructive migration as the production answer

## UI flow
Home:
- Resume Shoe when an active session exists
- Start New Shoe
- Review
- Settings

New Shoe:
- count selector: KO / KISS III
- deck selector: 1/2/4/6/8
- Fresh vs Join Mid-Shoe
- Start Viper
- defaults: KO, 6 decks, Fresh

Live HUD:
- camera preview occupies most of screen
- selected count mode
- large running count
- strategy-appropriate reference values
- penetration / deck context
- rolling recent-card strip
- Review count
- + Card
- Undo
- overflow: pause, freeze, vision debug, shoe info, new shoe, end session

Vision Debug overlay later exposes detector/recognizer/tracker rates, active tracks, locked/pending counts, ROI, and thermal state. V0.1 may show stable placeholders until real inference exists.

Review history:
- completed sessions
- card history
- corrections
- hard cases

## V0.1 scope
V0.1 is a foundation build, not final computer vision.

Must include:
- compilable Android app
- real CameraX preview and permission handling if stable
- Home/New Shoe/Live HUD/Settings/Review-shell navigation
- KO and KISS III pure engines
- Room schema and active-session recovery
- append-only ledger + resolver
- basic manual add, correction/invalidation, and undo plumbing
- CardDetector/CardRecognizer interfaces with fake/no-op implementation
- basic TrackedCard lifecycle/domain model and exactly-once guard tests
- GitHub Actions unit test + lint + debug APK build
- README with install/download instructions

May defer:
- trained neural detector/recognizer
- real perspective rectification
- dynamic ROI implementation
- automatic shuffle recognition
- discard-tray estimation
- sophisticated physical multi-object tracker
- video-file inference
- model hard-case export UI

## Verification requirements
Automated tests must cover at minimum:
- KO tag values and IRCs
- KISS III black/red 2 distinction and other tag values
- known sequences through both strategies
- full-deck/shoe invariants appropriate to each unbalanced strategy
- ledger correction/invalidation/manual-add resolution
- replay after correction
- KO -> KISS III -> KO round trip without ledger mutation
- same physical tracked card observed repeatedly commits exactly once
- temporary occlusion/reacquisition does not duplicate
- genuinely new identical card can create a new event
- face-down then reveal commits once
- low-confidence candidate remains pending and does not change count
- KISS III 2 with unknown color is held until color resolves
- crash-recovery reconstruction yields same derived state
- Room transaction behavior

CI commands:
- `./gradlew test`
- `./gradlew lintDebug`
- `./gradlew assembleDebug`

Artifact name:
- `CardViper-debug-apk`

## V0.1 success condition
From a phone, the user can open GitHub Actions, download the latest `CardViper-debug-apk`, install it on the Pixel 7, launch CardViper, create a 6-deck KO shoe, see a stable live camera/HUD, switch to KISS III without losing ledger state, exercise manual card/correction/undo behavior, close/reopen the app and recover the active shoe, and see all CI tests green.