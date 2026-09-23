package com.cardviper.app.data

import com.cardviper.app.model.CardLedgerEvent
import com.cardviper.app.session.ShoeSession
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf

interface SessionRepository {
    suspend fun getActiveSession(): ShoeSession?
    suspend fun getSession(sessionId: String): ShoeSession?
    suspend fun getEvents(sessionId: String): List<CardLedgerEvent>
    suspend fun getPendingReviewCount(sessionId: String): Int

    fun observeActiveSession(): Flow<ShoeSession?>
    fun observeEvents(sessionId: String): Flow<List<CardLedgerEvent>>
    fun observePendingReviewCount(sessionId: String): Flow<Int>

    suspend fun upsertSession(session: ShoeSession)
    suspend fun appendEvent(event: CardLedgerEvent)

    suspend fun getPendingReviews(sessionId: String): List<PendingReview> = emptyList()
    fun observePendingReviews(sessionId: String): Flow<List<PendingReview>> = flowOf(emptyList())
    suspend fun upsertPendingReview(review: PendingReview) = Unit

    suspend fun appendEventWithHardCase(event: CardLedgerEvent, hardCase: HardCaseSample?) {
        appendEvent(event)
        hardCase?.let { insertHardCase(it) }
    }

    suspend fun appendEventAndResolvePending(
        event: CardLedgerEvent,
        reviewId: String,
        resolvedAtEpochMillis: Long,
        hardCase: HardCaseSample?,
    ) {
        appendEvent(event)
        getPendingReviews(event.sessionId).firstOrNull { it.reviewId == reviewId }?.let { review ->
            upsertPendingReview(
                review.copy(
                    state = PendingReviewState.MANUALLY_RESOLVED,
                    updatedAtEpochMillis = resolvedAtEpochMillis,
                ),
            )
        }
        hardCase?.let { insertHardCase(it) }
    }

    suspend fun insertHardCase(sample: HardCaseSample) = Unit
}
