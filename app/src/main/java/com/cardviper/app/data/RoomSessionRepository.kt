package com.cardviper.app.data

import androidx.room.withTransaction
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.data.db.CardLedgerEventEntity
import com.cardviper.app.data.db.CardViperDatabase
import com.cardviper.app.data.db.HardCaseSampleEntity
import com.cardviper.app.data.db.PendingReviewEntity
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
    override suspend fun getEvents(sessionId: String): List<CardLedgerEvent> = dao.getEvents(sessionId).map { it.toDomain() }
    override suspend fun getPendingReviewCount(sessionId: String): Int = dao.getPendingReviewCount(sessionId)
    override suspend fun getPendingReviews(sessionId: String): List<PendingReview> = dao.getPendingReviews(sessionId).map { it.toDomain() }

    override fun observeActiveSession(): Flow<ShoeSession?> = dao.observeActiveSession().map { it?.toDomain() }
    override fun observeEvents(sessionId: String): Flow<List<CardLedgerEvent>> = dao.observeEvents(sessionId).map { rows -> rows.map { it.toDomain() } }
    override fun observePendingReviewCount(sessionId: String): Flow<Int> = dao.observePendingReviewCount(sessionId)
    override fun observePendingReviews(sessionId: String): Flow<List<PendingReview>> = dao.observePendingReviews(sessionId).map { rows -> rows.map { it.toDomain() } }

    override suspend fun upsertSession(session: ShoeSession) { dao.upsertSession(session.toEntity()) }

    override suspend fun appendEvent(event: CardLedgerEvent) {
        database.withTransaction { dao.insertEvent(event.toEntity()) }
    }

    override suspend fun upsertPendingReview(review: PendingReview) {
        dao.upsertPendingReview(review.toEntity())
    }

    override suspend fun appendEventWithHardCase(event: CardLedgerEvent, hardCase: HardCaseSample?) {
        database.withTransaction {
            dao.insertEvent(event.toEntity())
            hardCase?.let { dao.insertHardCase(it.toEntity()) }
        }
    }

    override suspend fun appendEventAndResolvePending(
        event: CardLedgerEvent,
        reviewId: String,
        resolvedAtEpochMillis: Long,
        hardCase: HardCaseSample?,
    ) {
        database.withTransaction {
            dao.insertEvent(event.toEntity())
            dao.updatePendingReviewState(reviewId, PendingReviewState.MANUALLY_RESOLVED.name, resolvedAtEpochMillis)
            hardCase?.let { dao.insertHardCase(it.toEntity()) }
        }
    }

    override suspend fun insertHardCase(sample: HardCaseSample) {
        dao.insertHardCase(sample.toEntity())
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

private fun PendingReview.toEntity() = PendingReviewEntity(
    reviewId = reviewId,
    sessionId = sessionId,
    trackId = trackId,
    createdAtEpochMillis = createdAtEpochMillis,
    updatedAtEpochMillis = updatedAtEpochMillis,
    bestRank = bestCard?.rank?.name,
    bestSuit = bestCard?.suit?.name,
    bestColor = bestCard?.color?.name,
    rankConfidence = rankConfidence,
    suitConfidence = suitConfidence,
    colorConfidence = colorConfidence,
    cropReference = cropReference,
    state = state.name,
)

private fun PendingReviewEntity.toDomain(): PendingReview {
    val parsedSuit = bestSuit?.let(CardSuit::valueOf)
    val parsedColor = bestColor?.let(CardColor::valueOf)
    val card = bestRank?.let {
        PlayingCard(
            rank = CardRank.valueOf(it),
            suit = parsedSuit,
            colorHint = if (parsedSuit == null) parsedColor else null,
        )
    }
    return PendingReview(
        reviewId = reviewId,
        sessionId = sessionId,
        trackId = trackId,
        createdAtEpochMillis = createdAtEpochMillis,
        updatedAtEpochMillis = updatedAtEpochMillis,
        bestCard = card,
        rankConfidence = rankConfidence,
        suitConfidence = suitConfidence,
        colorConfidence = colorConfidence,
        cropReference = cropReference,
        state = PendingReviewState.valueOf(state),
    )
}

private fun HardCaseSample.toEntity() = HardCaseSampleEntity(
    sampleId = sampleId,
    sessionId = sessionId,
    ledgerEventId = ledgerEventId,
    reviewId = reviewId,
    createdAtEpochMillis = createdAtEpochMillis,
    sampleType = sampleType.name,
    predictedRank = predictedCard?.rank?.name,
    actualRank = actualCard?.rank?.name,
    predictedSuit = predictedCard?.suit?.name,
    actualSuit = actualCard?.suit?.name,
    predictedColor = predictedCard?.color?.name,
    actualColor = actualCard?.color?.name,
    cropReference = cropReference,
    exported = exported,
)
