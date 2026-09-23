package com.cardviper.app.session

import com.cardviper.app.blackjack.CountStrategy
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.blackjack.Kiss3Strategy
import com.cardviper.app.blackjack.KoStrategy
import com.cardviper.app.blackjack.LedgerResolver
import com.cardviper.app.data.SessionRepository
import com.cardviper.app.model.CardEventSource
import com.cardviper.app.model.CardLedgerEvent
import com.cardviper.app.model.LedgerEventType
import com.cardviper.app.model.PlayingCard
import java.util.UUID
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.map

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
        repository.getActiveSession()?.let { previous ->
            repository.upsertSession(previous.copy(state = SessionState.ENDED, endedAtEpochMillis = clock()))
        }
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

    suspend fun effectiveCards(): List<com.cardviper.app.blackjack.ResolvedCard> {
        val session = repository.getActiveSession() ?: return emptyList()
        return resolver.resolve(repository.getEvents(session.sessionId))
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    fun observeEffectiveCards(): Flow<List<com.cardviper.app.blackjack.ResolvedCard>> =
        repository.observeActiveSession().flatMapLatest { session ->
            if (session == null) flowOf(emptyList())
            else repository.observeEvents(session.sessionId).map { resolver.resolve(it) }
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
        repository.upsertSession(
            current.copy(state = SessionState.ENDED, endedAtEpochMillis = clock()),
        )
    }

    private suspend fun updateState(state: SessionState) {
        val current = requireActiveSession()
        repository.upsertSession(current.copy(state = state))
    }

    private suspend fun requireActiveSession(): ShoeSession =
        requireNotNull(repository.getActiveSession()) { "No active CardViper shoe" }

    private suspend fun newEvent(
        sessionId: String,
        type: LedgerEventType,
        targetEventId: String? = null,
        card: PlayingCard? = null,
        source: CardEventSource,
    ): CardLedgerEvent {
        val nextSequence = (repository.getEvents(sessionId).maxOfOrNull { it.sequenceNumber } ?: 0L) + 1L
        return CardLedgerEvent(
            eventId = idFactory.newId(),
            sessionId = sessionId,
            sequenceNumber = nextSequence,
            timestampEpochMillis = clock(),
            eventType = type,
            targetEventId = targetEventId,
            card = card,
            source = source,
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
}
