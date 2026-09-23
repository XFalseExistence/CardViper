package com.cardviper.app.session

import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.data.SessionRepository
import com.cardviper.app.model.CardLedgerEvent
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.PlayingCard
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class SessionManagerTest {
    @Test
    fun freshSixDeckKoStartsAtMinusTwenty() = runBlocking {
        val repo = FakeSessionRepository()
        val manager = manager(repo)
        manager.startSession(CountStrategyId.KO, 6, StartMode.FRESH)
        val snapshot = manager.currentSnapshot()!!
        assertEquals(-20, snapshot.runningCount)
        assertEquals(0, snapshot.cardsSeen)
        assertEquals(0.0, snapshot.penetration, 0.00001)
    }

    @Test
    fun sixDeckKissThreeStartsAtNine() = runBlocking {
        val manager = manager(FakeSessionRepository())
        manager.startSession(CountStrategyId.KISS_III, 6, StartMode.FRESH)
        assertEquals(9, manager.currentSnapshot()!!.runningCount)
    }

    @Test
    fun switchingStrategyReplaysSameLedger() = runBlocking {
        val repo = FakeSessionRepository()
        val manager = manager(repo)
        val session = manager.startSession(CountStrategyId.KO, 6, StartMode.FRESH)
        manager.manualAdd(PlayingCard(CardRank.TWO, CardSuit.SPADES))
        manager.manualAdd(PlayingCard(CardRank.KING, CardSuit.HEARTS))
        val beforeEvents = repo.getEvents(session.sessionId)
        manager.switchStrategy(CountStrategyId.KISS_III)
        val kiss = manager.currentSnapshot()!!
        manager.switchStrategy(CountStrategyId.KO)
        val koAgain = manager.currentSnapshot()!!
        assertEquals(beforeEvents, repo.getEvents(session.sessionId))
        assertEquals(9, kiss.runningCount)
        assertEquals(-20, koAgain.runningCount)
    }

    @Test
    fun manualAddInvalidateAndPenetrationDeriveFromEffectiveLedger() = runBlocking {
        val manager = manager(FakeSessionRepository())
        manager.startSession(CountStrategyId.KO, 6, StartMode.FRESH)
        val added = manager.manualAdd(PlayingCard(CardRank.FIVE, CardSuit.CLUBS))
        val afterAdd = manager.currentSnapshot()!!
        assertEquals(-19, afterAdd.runningCount)
        assertEquals(1, afterAdd.cardsSeen)
        assertEquals(1.0 / 312.0, afterAdd.penetration, 0.00001)
        manager.invalidate(added.eventId)
        val afterInvalidate = manager.currentSnapshot()!!
        assertEquals(-20, afterInvalidate.runningCount)
        assertEquals(0, afterInvalidate.cardsSeen)
    }

    @Test
    fun pauseResumeAndEndPreserveFacts() = runBlocking {
        val manager = manager(FakeSessionRepository())
        manager.startSession(CountStrategyId.KO, 6, StartMode.FRESH)
        manager.manualAdd(PlayingCard(CardRank.SEVEN, CardSuit.CLUBS))
        manager.pause()
        assertEquals(SessionState.PAUSED, manager.currentSnapshot()!!.session.state)
        manager.resume()
        assertEquals(SessionState.ACTIVE, manager.currentSnapshot()!!.session.state)
        manager.end()
        assertEquals(null, manager.currentSnapshot())
    }

    @Test
    fun rebuildingManagerFromSameRepositoryReconstructsIdenticalSnapshot() = runBlocking {
        val repo = FakeSessionRepository()
        val first = manager(repo)
        first.startSession(CountStrategyId.KO, 6, StartMode.FRESH)
        first.manualAdd(PlayingCard(CardRank.FIVE, CardSuit.CLUBS))
        first.manualAdd(PlayingCard(CardRank.KING, CardSuit.HEARTS))
        val before = first.currentSnapshot()
        val recovered = manager(repo).currentSnapshot()
        assertNotNull(recovered)
        assertEquals(before, recovered)
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

private class FakeSessionRepository : SessionRepository {
    private val sessions = linkedMapOf<String, ShoeSession>()
    private val eventMap = linkedMapOf<String, MutableList<CardLedgerEvent>>()
    private val active = MutableStateFlow<ShoeSession?>(null)
    private val eventFlows = mutableMapOf<String, MutableStateFlow<List<CardLedgerEvent>>>()
    private val pendingCounts = mutableMapOf<String, MutableStateFlow<Int>>()

    override suspend fun getActiveSession(): ShoeSession? = active.value
    override suspend fun getSession(sessionId: String): ShoeSession? = sessions[sessionId]
    override suspend fun getEvents(sessionId: String): List<CardLedgerEvent> = eventMap[sessionId]?.toList().orEmpty()
    override suspend fun getPendingReviewCount(sessionId: String): Int = pendingCounts[sessionId]?.value ?: 0
    override fun observeActiveSession(): Flow<ShoeSession?> = active
    override fun observeEvents(sessionId: String): Flow<List<CardLedgerEvent>> = eventFlows.getOrPut(sessionId) { MutableStateFlow(getEventsBlocking(sessionId)) }
    override fun observePendingReviewCount(sessionId: String): Flow<Int> = pendingCounts.getOrPut(sessionId) { MutableStateFlow(0) }

    override suspend fun upsertSession(session: ShoeSession) {
        sessions[session.sessionId] = session
        if (session.state == SessionState.ENDED) {
            if (active.value?.sessionId == session.sessionId) active.value = null
        } else {
            active.value = session
        }
    }

    override suspend fun appendEvent(event: CardLedgerEvent) {
        eventMap.getOrPut(event.sessionId) { mutableListOf() }.add(event)
        eventFlows.getOrPut(event.sessionId) { MutableStateFlow(emptyList()) }.value = getEventsBlocking(event.sessionId)
    }

    private fun getEventsBlocking(sessionId: String) = eventMap[sessionId]?.toList().orEmpty()
}
