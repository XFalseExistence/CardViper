package com.cardviper.app.session

import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.data.HardCaseSample
import com.cardviper.app.data.HardCaseType
import com.cardviper.app.data.PendingReview
import com.cardviper.app.data.PendingReviewState
import com.cardviper.app.data.SessionRepository
import com.cardviper.app.model.CardLedgerEvent
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.LedgerEventType
import com.cardviper.app.model.PlayingCard
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CorrectionFlowTest {
    @Test
    fun correctionAppendsCorrectionAndHardCaseAndRecalculates() = runBlocking {
        val repo = CorrectionFakeRepository()
        val manager = manager(repo)
        manager.startSession(CountStrategyId.KO, 6, StartMode.FRESH)
        val original = manager.manualAdd(PlayingCard(CardRank.KING, CardSuit.HEARTS))
        manager.correctWithHardCase(original.eventId, PlayingCard(CardRank.FIVE, CardSuit.HEARTS))

        val events = repo.getEvents(original.sessionId)
        assertEquals(LedgerEventType.CARD_CORRECTED, events.last().eventType)
        assertEquals(-19, manager.currentSnapshot()!!.runningCount)
        assertEquals(HardCaseType.WRONG_RANK, repo.hardCases.single().sampleType)
    }

    @Test
    fun falsePositiveInvalidationCreatesHardCaseAndRemovesCardFromCount() = runBlocking {
        val repo = CorrectionFakeRepository()
        val manager = manager(repo)
        manager.startSession(CountStrategyId.KO, 6, StartMode.FRESH)
        val original = manager.manualAdd(PlayingCard(CardRank.FIVE, CardSuit.CLUBS))
        manager.invalidateAsFalsePositive(original.eventId)

        assertEquals(-20, manager.currentSnapshot()!!.runningCount)
        assertEquals(HardCaseType.FALSE_POSITIVE, repo.hardCases.single().sampleType)
    }

    @Test
    fun pendingReviewDoesNotCountUntilResolved() = runBlocking {
        val repo = CorrectionFakeRepository()
        val manager = manager(repo)
        val session = manager.startSession(CountStrategyId.KO, 6, StartMode.FRESH)
        repo.upsertPendingReview(
            PendingReview(
                reviewId = "review-1",
                sessionId = session.sessionId,
                trackId = 77L,
                createdAtEpochMillis = 1_000L,
                updatedAtEpochMillis = 1_000L,
                bestCard = PlayingCard(CardRank.THREE, CardSuit.HEARTS),
                rankConfidence = 0.58f,
                suitConfidence = 0.80f,
                colorConfidence = 0.95f,
                cropReference = "pending/review-1.webp",
                state = PendingReviewState.PENDING,
            ),
        )
        assertEquals(-20, manager.currentSnapshot()!!.runningCount)
        assertEquals(1, manager.currentSnapshot()!!.pendingReviews)

        manager.resolvePendingReview("review-1", PlayingCard(CardRank.THREE, CardSuit.HEARTS))

        assertEquals(-19, manager.currentSnapshot()!!.runningCount)
        assertEquals(0, manager.currentSnapshot()!!.pendingReviews)
        assertEquals(PendingReviewState.MANUALLY_RESOLVED, repo.pending.single().state)
        assertTrue(repo.hardCases.any { it.sampleType == HardCaseType.LOW_CONFIDENCE })
    }

    private fun manager(repo: SessionRepository) = SessionManager(
        repository = repo,
        idFactory = object : IdFactory {
            var next = 0
            override fun newId(): String = "id-${next++}"
        },
        clock = { 1_000L },
    )
}

private class CorrectionFakeRepository : SessionRepository {
    private val active = MutableStateFlow<ShoeSession?>(null)
    private val events = mutableListOf<CardLedgerEvent>()
    private val eventFlow = MutableStateFlow<List<CardLedgerEvent>>(emptyList())
    private val pendingFlow = MutableStateFlow<List<PendingReview>>(emptyList())
    val hardCases = mutableListOf<HardCaseSample>()
    val pending: List<PendingReview> get() = pendingFlow.value

    override suspend fun getActiveSession(): ShoeSession? = active.value
    override suspend fun getSession(sessionId: String): ShoeSession? = active.value?.takeIf { it.sessionId == sessionId }
    override suspend fun getEvents(sessionId: String): List<CardLedgerEvent> = events.filter { it.sessionId == sessionId }
    override suspend fun getPendingReviewCount(sessionId: String): Int = pendingFlow.value.count { it.sessionId == sessionId && it.state == PendingReviewState.PENDING }
    override suspend fun getPendingReviews(sessionId: String): List<PendingReview> = pendingFlow.value.filter { it.sessionId == sessionId }
    override fun observeActiveSession(): Flow<ShoeSession?> = active
    override fun observeEvents(sessionId: String): Flow<List<CardLedgerEvent>> = eventFlow
    override fun observePendingReviewCount(sessionId: String): Flow<Int> = MutableStateFlow(getPendingCountBlocking(sessionId))
    override fun observePendingReviews(sessionId: String): Flow<List<PendingReview>> = pendingFlow

    override suspend fun upsertSession(session: ShoeSession) { active.value = if (session.state == SessionState.ENDED) null else session }
    override suspend fun appendEvent(event: CardLedgerEvent) { events += event; eventFlow.value = events.toList() }
    override suspend fun upsertPendingReview(review: PendingReview) { pendingFlow.value = pendingFlow.value.filterNot { it.reviewId == review.reviewId } + review }
    override suspend fun appendEventAndResolvePending(event: CardLedgerEvent, reviewId: String, resolvedAtEpochMillis: Long, hardCase: HardCaseSample?) {
        appendEvent(event)
        pendingFlow.value = pendingFlow.value.map { if (it.reviewId == reviewId) it.copy(state = PendingReviewState.MANUALLY_RESOLVED, updatedAtEpochMillis = resolvedAtEpochMillis) else it }
        hardCase?.let(hardCases::add)
    }
    override suspend fun appendEventWithHardCase(event: CardLedgerEvent, hardCase: HardCaseSample?) {
        appendEvent(event)
        hardCase?.let(hardCases::add)
    }
    override suspend fun insertHardCase(sample: HardCaseSample) { hardCases += sample }

    private fun getPendingCountBlocking(sessionId: String) = pendingFlow.value.count { it.sessionId == sessionId && it.state == PendingReviewState.PENDING }
}
