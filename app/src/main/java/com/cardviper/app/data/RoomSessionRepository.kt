package com.cardviper.app.data

import androidx.room.withTransaction
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.data.db.CardLedgerEventEntity
import com.cardviper.app.data.db.CardViperDatabase
import com.cardviper.app.data.db.ShoeSessionEntity
import com.cardviper.app.model.CardColor
import com.cardviper.app.model.CardEventSource
import com.cardviper.app.model.CardLedgerEvent
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.LedgerEventType
import com.cardviper.app.model.PlayingCard
import com.cardviper.app.session.SessionState
import com.cardviper.app.session.ShoeSession
import com.cardviper.app.session.StartMode
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

class RoomSessionRepository(
    private val database: CardViperDatabase,
) : SessionRepository {
    private val dao = database.dao()

    override suspend fun getActiveSession(): ShoeSession? = dao.getActiveSession()?.toDomain()

    override suspend fun getSession(sessionId: String): ShoeSession? = dao.getSession(sessionId)?.toDomain()

    override suspend fun getEvents(sessionId: String): List<CardLedgerEvent> =
        dao.getEvents(sessionId).map { it.toDomain() }

    override suspend fun getPendingReviewCount(sessionId: String): Int = dao.getPendingReviewCount(sessionId)

    override fun observeActiveSession(): Flow<ShoeSession?> = dao.observeActiveSession().map { it?.toDomain() }

    override fun observeEvents(sessionId: String): Flow<List<CardLedgerEvent>> =
        dao.observeEvents(sessionId).map { rows -> rows.map { it.toDomain() } }

    override fun observePendingReviewCount(sessionId: String): Flow<Int> = dao.observePendingReviewCount(sessionId)

    override suspend fun upsertSession(session: ShoeSession) {
        dao.upsertSession(session.toEntity())
    }

    override suspend fun appendEvent(event: CardLedgerEvent) {
        database.withTransaction {
            dao.insertEvent(event.toEntity())
        }
    }

    suspend fun appendEventAndResolvePending(
        event: CardLedgerEvent,
        reviewId: String,
        resolvedAtEpochMillis: Long,
    ) {
        database.withTransaction {
            dao.insertEvent(event.toEntity())
            dao.updatePendingReviewState(reviewId, "MANUALLY_RESOLVED", resolvedAtEpochMillis)
        }
    }
}

private fun ShoeSession.toEntity() = ShoeSessionEntity(
    sessionId = sessionId,
    countStrategy = countStrategy.name,
    strategyVersion = strategyVersion,
    nominalDecks = nominalDecks,
    startMode = startMode.name,
    startingRunningCount = startingRunningCount,
    startingDeckEstimate = startingDeckEstimate,
    state = state.name,
    createdAtEpochMillis = createdAtEpochMillis,
    startedAtEpochMillis = startedAtEpochMillis,
    endedAtEpochMillis = endedAtEpochMillis,
)

private fun ShoeSessionEntity.toDomain() = ShoeSession(
    sessionId = sessionId,
    countStrategy = CountStrategyId.valueOf(countStrategy),
    strategyVersion = strategyVersion,
    nominalDecks = nominalDecks,
    startMode = StartMode.valueOf(startMode),
    startingRunningCount = startingRunningCount,
    startingDeckEstimate = startingDeckEstimate,
    state = SessionState.valueOf(state),
    createdAtEpochMillis = createdAtEpochMillis,
    startedAtEpochMillis = startedAtEpochMillis,
    endedAtEpochMillis = endedAtEpochMillis,
)

private fun CardLedgerEvent.toEntity() = CardLedgerEventEntity(
    eventId = eventId,
    sessionId = sessionId,
    sequenceNumber = sequenceNumber,
    timestampEpochMillis = timestampEpochMillis,
    eventType = eventType.name,
    targetEventId = targetEventId,
    trackId = trackId,
    rank = card?.rank?.name,
    suit = card?.suit?.name,
    colorHint = card?.colorHint?.name,
    source = source.name,
    rankConfidence = rankConfidence,
    colorConfidence = colorConfidence,
    suitConfidence = suitConfidence,
    cropReference = cropReference,
)

private fun CardLedgerEventEntity.toDomain(): CardLedgerEvent {
    val parsedSuit = suit?.let(CardSuit::valueOf)
    val parsedColor = colorHint?.let(CardColor::valueOf)
    val parsedCard = rank?.let { PlayingCard(CardRank.valueOf(it), parsedSuit, parsedColor) }
    return CardLedgerEvent(
        eventId = eventId,
        sessionId = sessionId,
        sequenceNumber = sequenceNumber,
        timestampEpochMillis = timestampEpochMillis,
        eventType = LedgerEventType.valueOf(eventType),
        targetEventId = targetEventId,
        trackId = trackId,
        card = parsedCard,
        source = CardEventSource.valueOf(source),
        rankConfidence = rankConfidence,
        colorConfidence = colorConfidence,
        suitConfidence = suitConfidence,
        cropReference = cropReference,
    )
}
