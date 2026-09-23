package com.cardviper.app.ui

import androidx.lifecycle.ViewModelStore
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.data.SessionRepository
import com.cardviper.app.model.*
import com.cardviper.app.session.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.test.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class CardViperViewModelTest {
    private val dispatcher = StandardTestDispatcher()
    private lateinit var repo: UiFakeRepository
    private lateinit var manager: SessionManager
    private lateinit var model: CardViperViewModel
    private val store = ViewModelStore()
    @Before fun setup() {
        Dispatchers.setMain(dispatcher)
        repo = UiFakeRepository()
        manager = SessionManager(repo)
        model = CardViperViewModel(manager, repo)
        store.put("model", model)
    }
    @After fun tearDown() { store.clear(); Dispatchers.resetMain() }

    @Test fun defaultsAndSessionAreIndependentOfCameraPermission() = runTest(dispatcher) {
        model.startShoe(); advanceUntilIdle()
        val snapshot = model.uiState.value.snapshot!!
        assertEquals(CountStrategyId.KO, snapshot.session.countStrategy)
        assertEquals(6, snapshot.session.nominalDecks)
        assertEquals(StartMode.FRESH, snapshot.session.startMode)
        assertEquals(-20, snapshot.runningCount)
        assertEquals(0, snapshot.cardsSeen)
    }
    @Test fun switchingModesReplaysSameEventsAndUndoAppendsInvalidation() = runTest(dispatcher) {
        model.startShoe(); advanceUntilIdle()
        model.manualAdd(PlayingCard(CardRank.TWO, CardSuit.SPADES)); advanceUntilIdle()
        val sessionId = repo.getActiveSession()!!.sessionId
        val before = repo.getEvents(sessionId)
        model.switchStrategy(CountStrategyId.KISS_III); advanceUntilIdle()
        assertEquals(10, model.uiState.value.snapshot!!.runningCount)
        model.switchStrategy(CountStrategyId.KO); advanceUntilIdle()
        assertEquals(-19, model.uiState.value.snapshot!!.runningCount)
        assertEquals(before, repo.getEvents(sessionId))
        model.undo(); advanceUntilIdle()
        assertEquals(-20, model.uiState.value.snapshot!!.runningCount)
        assertEquals(2, repo.getEvents(sessionId).size)
        assertEquals(LedgerEventType.CARD_INVALIDATED, repo.getEvents(sessionId).last().eventType)
    }
    @Test fun kissRequiresColorForManualTwoAndRejectsUnsupportedDecks() = runTest(dispatcher) {
        model.startShoe(CountStrategyId.KISS_III); advanceUntilIdle()
        model.manualAdd(PlayingCard(CardRank.TWO)); advanceUntilIdle()
        assertNotNull(model.uiState.value.errorMessage)
        assertEquals(0, model.uiState.value.snapshot!!.cardsSeen)
        model.manualAdd(PlayingCard(CardRank.TWO, colorHint = CardColor.RED)); advanceUntilIdle()
        assertEquals(9, model.uiState.value.snapshot!!.runningCount)
        model.startShoe(CountStrategyId.KISS_III, 2); advanceUntilIdle()
        assertNotNull(model.uiState.value.errorMessage)
        assertEquals(1, model.uiState.value.snapshot!!.cardsSeen)
    }
    @Test fun backgroundPauseResumeAndRecreatedModelKeepShoe() = runTest(dispatcher) {
        model.startShoe(); advanceUntilIdle()
        model.manualAdd(PlayingCard(CardRank.SEVEN)); advanceUntilIdle()
        val before = model.uiState.value.snapshot!!
        model.onAppBackgrounded(); advanceUntilIdle()
        assertEquals(before, model.uiState.value.snapshot)
        model.pause(); advanceUntilIdle()
        assertEquals(SessionState.PAUSED, model.uiState.value.snapshot!!.session.state)
        model.resume(); advanceUntilIdle()
        val recreated = CardViperViewModel(SessionManager(repo), repo)
        store.put("recreated", recreated); advanceUntilIdle()
        assertEquals(before, recreated.uiState.value.snapshot)
    }
    @Test fun rapidManualActionsAreSerializedWithUniqueSequences() = runTest(dispatcher) {
        model.startShoe(); advanceUntilIdle()
        repeat(10) { model.manualAdd(PlayingCard(CardRank.FIVE)) }; advanceUntilIdle()
        val events = repo.getEvents(repo.getActiveSession()!!.sessionId)
        assertEquals((1L..10L).toList(), events.map { it.sequenceNumber })
        assertEquals(-10, model.uiState.value.snapshot!!.runningCount)
    }
    @Test fun invalidMidShoeDoesNotReplaceCurrentSessionAndNavigationWaitsForSuccess() = runTest(dispatcher) {
        var started = false
        model.startShoe(onStarted = { started = true }); advanceUntilIdle()
        assertTrue(started)
        val before = model.uiState.value.snapshot
        model.startShoe(startMode = StartMode.MID_SHOE, startingRunningCount = null); advanceUntilIdle()
        assertEquals(before, model.uiState.value.snapshot)
        assertNotNull(model.uiState.value.errorMessage)
    }
    @Test fun unknownTwoCannotSilentlyDisappearWhenSwitchingToKiss() = runTest(dispatcher) {
        model.startShoe(); advanceUntilIdle()
        model.manualAdd(PlayingCard(CardRank.TWO)); advanceUntilIdle()
        val sessionId = repo.getActiveSession()!!.sessionId
        val ledger = repo.getEvents(sessionId)
        model.switchStrategy(CountStrategyId.KISS_III); advanceUntilIdle()
        assertEquals(CountStrategyId.KO, model.uiState.value.snapshot!!.session.countStrategy)
        assertEquals(-19, model.uiState.value.snapshot!!.runningCount)
        assertEquals(ledger, repo.getEvents(sessionId))
        assertNotNull(model.uiState.value.errorMessage)
    }
    @Test fun unsupportedMidShoeModeSwitchAndBadEstimatePreserveCurrentShoe() = runTest(dispatcher) {
        model.startShoe(decks = 2, startMode = StartMode.MID_SHOE,
            startingRunningCount = 3, startingDeckEstimate = 1.5); advanceUntilIdle()
        val before = model.uiState.value.snapshot
        model.switchStrategy(CountStrategyId.KISS_III); advanceUntilIdle()
        assertEquals(before, model.uiState.value.snapshot)
        model.startShoe(startMode = StartMode.MID_SHOE, startingRunningCount = 2,
            startingDeckEstimate = Double.NaN); advanceUntilIdle()
        assertEquals(before, model.uiState.value.snapshot)
    }
    @Test fun endingShoePreservesItsLedgerAndClearsResumeState() = runTest(dispatcher) {
        model.startShoe(); advanceUntilIdle()
        model.manualAdd(PlayingCard(CardRank.KING)); advanceUntilIdle()
        val id = repo.getActiveSession()!!.sessionId
        var navigated = false
        model.endShoe { navigated = true }; advanceUntilIdle()
        assertTrue(navigated)
        assertNull(model.uiState.value.snapshot)
        assertEquals(SessionState.ENDED, repo.getSession(id)!!.state)
        assertEquals(1, repo.getEvents(id).size)
    }
    @Test fun joinedShoeCannotReinterpretAStartCountFromAnotherSystem() = runTest(dispatcher) {
        model.startShoe(startMode = StartMode.MID_SHOE, startingRunningCount = -10); advanceUntilIdle()
        model.manualAdd(PlayingCard(CardRank.FIVE)); advanceUntilIdle()
        val before = model.uiState.value.snapshot
        model.switchStrategy(CountStrategyId.KISS_III); advanceUntilIdle()
        assertEquals(before, model.uiState.value.snapshot)
        assertNotNull(model.uiState.value.errorMessage)
    }
}

private class UiFakeRepository : SessionRepository {
    private val sessions = linkedMapOf<String, ShoeSession>()
    private val active = MutableStateFlow<ShoeSession?>(null)
    private val events = mutableMapOf<String, MutableStateFlow<List<CardLedgerEvent>>>()
    private fun eventFlow(id: String) = events.getOrPut(id) { MutableStateFlow(emptyList()) }
    override suspend fun getActiveSession() = active.value
    override suspend fun getSession(sessionId: String) = sessions[sessionId]
    override suspend fun getEvents(sessionId: String) = eventFlow(sessionId).value
    override suspend fun getPendingReviewCount(sessionId: String) = 0
    override fun observeActiveSession(): Flow<ShoeSession?> = active
    override fun observeEvents(sessionId: String): Flow<List<CardLedgerEvent>> = eventFlow(sessionId)
    override fun observePendingReviewCount(sessionId: String): Flow<Int> = flowOf(0)
    override suspend fun upsertSession(session: ShoeSession) {
        sessions[session.sessionId] = session
        active.value = if(session.state == SessionState.ENDED) null else session
    }
    override suspend fun appendEvent(event: CardLedgerEvent) {
        eventFlow(event.sessionId).value += event
    }
}
