package com.cardviper.app.session

import com.cardviper.app.blackjack.CountStrategy
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.blackjack.Kiss3Strategy
import com.cardviper.app.blackjack.KoStrategy
import com.cardviper.app.blackjack.LedgerResolver
import com.cardviper.app.blackjack.ResolvedCard
import com.cardviper.app.data.HardCaseSample
import com.cardviper.app.data.HardCaseType
import com.cardviper.app.data.PendingReview
import com.cardviper.app.data.PendingReviewState
import com.cardviper.app.data.SessionRepository
import com.cardviper.app.model.CardEventSource
import com.cardviper.app.model.CardLedgerEvent
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.LedgerEventType
import com.cardviper.app.model.PlayingCard
import java.util.UUID
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf

fun interface IdFactory {
    fun newId(): String
}

class SessionManager(
    private val repository: SessionRepository,
    private val idFactory: IdFactory = IdFactory { UUID.randomUUID().toString() },
    private val clock: () -> Long = { System.currentTimeMillis() },
    private val resolver: LedgerResolver = LedgerResolver(),
) {
    suspend fun startSession(
        strategyId: CountStrategyId = CountStrategyId.KO,
        nominalDecks: Int = 6,
        startMode: StartMode = StartMode.FRESH,
        startingRunningCount: Int? = null,
        startingDeckEstimate: Double? = null,
    ): ShoeSession {
        require(nominalDecks > 0) { "nominalDecks must be positive" }
        val strategy = strategy(strategyId)
        val startCount = when (startMode) {
            StartMode.FRESH -> strategy.initialRunningCount(nominalDecks)
            StartMode.MID_SHOE -> requireNotNull(startingRunningCount) {
                "MID_SHOE requires a known startingRunningCount in V0.1"
            }
        }
        val now = clock()
        val session = ShoeSession(
            sessionId = idFactory.newId(),
            countStrategy = strategyId,
            strategyVersion = strategy.version,
            nominalDecks = nominalDecks,
            startMode = startMode,
            startingRunningCount = startCount,
            startingDeckEstimate = startingDeckEstimate,
            state = SessionState.ACTIVE,
            createdAtEpochMillis = now,
            startedAtEpochMillis = now,
        )
        repository.upsertSession(session)
        return session
    }

    suspend fun currentSnapshot(): SessionSnapshot? {
        val session = repository.getActiveSession() ?: return null
        return calculateSnapshot(
            session = session,
            events = repository.getEvents(session.sessionId),
            pendingReviews = repository.getPendingReviewCount(session.sessionId),
        )
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    fun observeSnapshot(): Flow<SessionSnapshot?> = repository.observeActiveSession().flatMapLatest { session ->
        if (session == null) {
            flowOf(null)
        } else {
            combine(
                repository.observeEvents(session.sessionId),
                repository.observePendingReviewCount(session.sessionId),
            ) { events, pending -> calculateSnapshot(session, events, pending) }
        }
    }

    suspend fun switchStrategy(strategyId: CountStrategyId): ShoeSession {
        val current = requireActiveSession()
        val next = strategy(strategyId)
        val updated = current.copy(
            countStrategy = strategyId,
            strategyVersion = next.version,
            startingRunningCount = when (current.startMode) {
                StartMode.FRESH -> next.initialRunningCount(current.nominalDecks)
                StartMode.MID_SHOE -> current.startingRunningCount
            },
        )
        repository.upsertSession(updated)
        return updated
    }

    suspend fun manualAdd(card: PlayingCard): CardLedgerEvent {
        val session = requireActiveSession()
        val event = newEvent(
            sessionId = session.sessionId,
            type = LedgerEventType.CARD_MANUAL_ADDED,
            card = card,
            source = CardEventSource.MANUAL,
        )
        repository.appendEvent(event)
        return event
    }

    suspend fun invalidate(targetEventId: String): CardLedgerEvent {
        val session = requireActiveSession()
        val event = newEvent(
            sessionId = session.sessionId,
            type = LedgerEventType.CARD_INVALIDATED,
            targetEventId = targetEventId,
            source = CardEventSource.MANUAL,
        )
        repository.appendEvent(event)
        return event
    }

    suspend fun correct(targetEventId: String, card: PlayingCard): CardLedgerEvent {
        val session = requireActiveSession()
        val event = newEvent(
            sessionId = session.sessionId,
            type = LedgerEventType.CARD_CORRECTED,
            targetEventId = targetEventId,
            card = card,
            source = CardEventSource.CORRECTION,
        )
        repository.appendEvent(event)
        return event
    }

    suspend fun correctWithHardCase(targetEventId: String, card: PlayingCard): CardLedgerEvent {
        val session = requireActiveSession()
        val context = targetContext(session, targetEventId)
        require(context.resolved.card != card) { "Corrected card must differ from the current card" }
        val event = newEvent(
            sessionId = session.sessionId,
            type = LedgerEventType.CARD_CORRECTED,
            targetEventId = context.resolved.effectiveEventId,
            card = card,
            source = CardEventSource.CORRECTION,
        )
        val hardCase = HardCaseSample(
            sampleId = idFactory.newId(),
            sessionId = session.sessionId,
            ledgerEventId = event.eventId,
            createdAtEpochMillis = clock(),
            sampleType = correctionType(context.resolved.card, card),
            predictedCard = context.resolved.card,
            actualCard = card,
            cropReference = context.effectiveEvent?.cropReference,
        )
        repository.appendEventWithHardCase(event, hardCase)
        return event
    }

    suspend fun invalidateAsFalsePositive(targetEventId: String): CardLedgerEvent {
        val session = requireActiveSession()
        val context = targetContext(session, targetEventId)
        val event = newEvent(
            sessionId = session.sessionId,
            type = LedgerEventType.CARD_INVALIDATED,
            targetEventId = context.resolved.effectiveEventId,
            source = CardEventSource.CORRECTION,
        )
        val hardCase = HardCaseSample(
            sampleId = idFactory.newId(),
            sessionId = session.sessionId,
            ledgerEventId = event.eventId,
            createdAtEpochMillis = clock(),
            sampleType = HardCaseType.FALSE_POSITIVE,
            predictedCard = context.resolved.card,
            cropReference = context.effectiveEvent?.cropReference,
        )
        repository.appendEventWithHardCase(event, hardCase)
        return event
    }

    suspend fun resolvePendingReview(reviewId: String, card: PlayingCard): CardLedgerEvent {
        val session = requireActiveSession()
        val review = requirePendingReview(session, reviewId)
        require(session.countStrategy != CountStrategyId.KISS_III || card.rank != CardRank.TWO || card.color != null) {
            "KISS III requires the color of a 2 before it can be counted"
        }
        val event = newEvent(
            sessionId = session.sessionId,
            type = LedgerEventType.CARD_COMMITTED,
            card = card,
            source = CardEventSource.CORRECTION,
            trackId = review.trackId,
            rankConfidence = review.rankConfidence,
            colorConfidence = review.colorConfidence,
            suitConfidence = review.suitConfidence,
            cropReference = review.cropReference,
        )
        val hardCase = HardCaseSample(
            sampleId = idFactory.newId(),
            sessionId = session.sessionId,
            ledgerEventId = event.eventId,
            reviewId = review.reviewId,
            createdAtEpochMillis = clock(),
            sampleType = HardCaseType.LOW_CONFIDENCE,
            predictedCard = review.bestCard,
            actualCard = card,
            cropReference = review.cropReference,
        )
        repository.appendEventAndResolvePending(event, review.reviewId, clock(), hardCase)
        return event
    }

    suspend fun discardPendingReview(reviewId: String) {
        val session = requireActiveSession()
        val review = requirePendingReview(session, reviewId)
        val now = clock()
        repository.upsertPendingReview(review.copy(state = PendingReviewState.DISCARDED, updatedAtEpochMillis = now))
        repository.insertHardCase(
            HardCaseSample(
                sampleId = idFactory.newId(),
                sessionId = session.sessionId,
                reviewId = review.reviewId,
                createdAtEpochMillis = now,
                sampleType = HardCaseType.FALSE_POSITIVE,
                predictedCard = review.bestCard,
                cropReference = review.cropReference,
            ),
        )
    }

    suspend fun undoLast(): CardLedgerEvent? {
        val session = requireActiveSession()
        val effective = resolver.resolve(repository.getEvents(session.sessionId))
        val latest = effective.lastOrNull() ?: return null
        return invalidate(latest.effectiveEventId)
    }

    suspend fun pause() = updateState(SessionState.PAUSED)
    suspend fun resume() = updateState(SessionState.ACTIVE)

    suspend fun end() {
        val current = requireActiveSession()
        repository.upsertSession(current.copy(state = SessionState.ENDED, endedAtEpochMillis = clock()))
    }

    private suspend fun updateState(state: SessionState) {
        val current = requireActiveSession()
        repository.upsertSession(current.copy(state = state))
    }

    private suspend fun requireActiveSession(): ShoeSession =
        requireNotNull(repository.getActiveSession()) { "No active CardViper shoe" }

    private suspend fun requirePendingReview(session: ShoeSession, reviewId: String): PendingReview =
        requireNotNull(repository.getPendingReviews(session.sessionId).firstOrNull {
            it.reviewId == reviewId && it.state == PendingReviewState.PENDING
        }) { "Pending review not found" }

    private suspend fun targetContext(session: ShoeSession, targetEventId: String): TargetContext {
        val events = repository.getEvents(session.sessionId)
        val resolved = requireNotNull(
            resolver.resolve(events).firstOrNull {
                it.rootEventId == targetEventId || it.effectiveEventId == targetEventId
            },
        ) { "Card is no longer active in the ledger" }
        return TargetContext(resolved, events.firstOrNull { it.eventId == resolved.effectiveEventId })
    }

    private fun correctionType(before: PlayingCard, after: PlayingCard): HardCaseType = when {
        before.rank != after.rank -> HardCaseType.WRONG_RANK
        before.suit != after.suit -> HardCaseType.WRONG_SUIT
        before.color != after.color -> HardCaseType.WRONG_COLOR
        else -> HardCaseType.WRONG_RANK
    }

    private suspend fun newEvent(
        sessionId: String,
        type: LedgerEventType,
        targetEventId: String? = null,
        card: PlayingCard? = null,
        source: CardEventSource,
        trackId: Long? = null,
        rankConfidence: Float? = null,
        colorConfidence: Float? = null,
        suitConfidence: Float? = null,
        cropReference: String? = null,
    ): CardLedgerEvent {
        val nextSequence = (repository.getEvents(sessionId).maxOfOrNull { it.sequenceNumber } ?: 0L) + 1L
        return CardLedgerEvent(
            eventId = idFactory.newId(),
            sessionId = sessionId,
            sequenceNumber = nextSequence,
            timestampEpochMillis = clock(),
            eventType = type,
            targetEventId = targetEventId,
            trackId = trackId,
            card = card,
            source = source,
            rankConfidence = rankConfidence,
            colorConfidence = colorConfidence,
            suitConfidence = suitConfidence,
            cropReference = cropReference,
        )
    }

    private fun calculateSnapshot(
        session: ShoeSession,
        events: List<CardLedgerEvent>,
        pendingReviews: Int,
    ): SessionSnapshot {
        val strategy = strategy(session.countStrategy)
        val resolved = resolver.resolve(events)
        val count = strategy.evaluate(session.startingRunningCount, resolved.map { it.card })
        val baseDecks = session.startingDeckEstimate ?: session.nominalDecks.toDouble()
        val decksRemaining = (baseDecks - (resolved.size / 52.0)).coerceAtLeast(0.0)
        val penetration = when (session.startMode) {
            StartMode.FRESH -> resolved.size / (session.nominalDecks * 52.0)
            StartMode.MID_SHOE -> ((session.nominalDecks - decksRemaining) / session.nominalDecks).coerceIn(0.0, 1.0)
        }
        return SessionSnapshot(
            session = session,
            runningCount = count.runningCount,
            cardsSeen = resolved.size,
            penetration = penetration,
            estimatedDecksRemaining = decksRemaining,
            keyCount = strategy.keyCount(session.nominalDecks),
            insuranceCount = strategy.insuranceCount(session.nominalDecks),
            pendingReviews = pendingReviews,
        )
    }

    private fun strategy(id: CountStrategyId): CountStrategy = when (id) {
        CountStrategyId.KO -> KoStrategy()
        CountStrategyId.KISS_III -> Kiss3Strategy()
    }

    private data class TargetContext(
        val resolved: ResolvedCard,
        val effectiveEvent: CardLedgerEvent?,
    )
}
