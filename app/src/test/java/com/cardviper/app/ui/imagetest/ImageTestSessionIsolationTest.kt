package com.cardviper.app.ui.imagetest

import androidx.lifecycle.ViewModelStore
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.data.SessionRepository
import com.cardviper.app.model.CardLedgerEvent
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.PlayingCard
import com.cardviper.app.session.SessionManager
import com.cardviper.app.session.ShoeSession
import com.cardviper.app.session.StartMode
import com.cardviper.app.vision.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class ImageTestSessionIsolationTest {
    private val dispatcher = StandardTestDispatcher()
    private val store = ViewModelStore()
    @Before fun setup() { Dispatchers.setMain(dispatcher) }
    @After fun tearDown() { store.clear(); Dispatchers.resetMain() }

    @Test fun imagePreviewCannotChangeActiveShoeEventsOrSnapshot() = runTest(dispatcher) {
        val engine = object : ImageRecognitionEngine {
            override suspend fun recognize(image: VisionImage, onProgress: (RecognitionProgress) -> Unit): ImageRecognitionResult {
                onProgress(RecognitionProgress(RecognitionStage.DETECTING))
                onProgress(RecognitionProgress(RecognitionStage.CLASSIFYING, 1, 1))
                val card = ImageCardObservation(CardCandidate(0f, 0f, 1f, 1f, 1f),
                    PixelRect(0, 0, 1, 1), CardLabelCodec.decode("2C"), 0.55f, true)
                return ImageRecognitionResult(listOf(card), 1, 1, 1, 0)
            }
        }
        val state = analyzeWithActiveShoe(engine, ImageTestPhase.COMPLETE)
        assertEquals(1, state.result!!.koDelta)
        assertEquals(1, state.result.kiss3Delta)
    }

    @Test fun preModelEngineReportsMissingModelWithoutChangingShoe() = runTest(dispatcher) {
        val engine = DefaultImageRecognitionEngine(UnavailableCardDetector(), NoOpCardRecognizer())
        val state = analyzeWithActiveShoe(engine, ImageTestPhase.ERROR)
        assertEquals("VISION MODEL NOT INSTALLED", state.statusText)
        assertNull(state.result)
        assertNotNull(state.image)
    }

    private suspend fun analyzeWithActiveShoe(engine: ImageRecognitionEngine, phase: ImageTestPhase): ImageRecognitionUiState {
        val repository = IsolationRepository()
        val manager = SessionManager(repository)
        val session = manager.startSession(CountStrategyId.KO, 6, StartMode.FRESH)
        manager.manualAdd(PlayingCard(CardRank.KING, CardSuit.SPADES))
        val eventsBefore = repository.getEvents(session.sessionId)
        val snapshotBefore = manager.currentSnapshot()
        assertEquals(1, eventsBefore.size)
        assertEquals(-21, snapshotBefore!!.runningCount)

        val image = VisionImage(1, 1, byteArrayOf(10, 20, 30))
        // This construction must stay independent of every shoe/session/repository dependency.
        val model = ImageRecognitionViewModel(ImageSourceLoader { image }, engine)
        store.put("image-test", model)
        model.analyze("content://test/image")
        val state = model.uiState.first { it.phase == phase }
        assertSame(image, state.image)
        assertEquals(eventsBefore, repository.getEvents(session.sessionId))
        assertEquals(snapshotBefore, manager.currentSnapshot())
        assertEquals(session, repository.getActiveSession())
        return state
    }
}

private class IsolationRepository : SessionRepository {
    private var session: ShoeSession? = null
    private val events = mutableListOf<CardLedgerEvent>()
    override suspend fun getActiveSession() = session
    override suspend fun getSession(sessionId: String) = session?.takeIf { it.sessionId == sessionId }
    override suspend fun getEvents(sessionId: String) = events.filter { it.sessionId == sessionId }
    override suspend fun getPendingReviewCount(sessionId: String) = 0
    override fun observeActiveSession() = flowOf(session)
    override fun observeEvents(sessionId: String) = flowOf(events.filter { it.sessionId == sessionId })
    override fun observePendingReviewCount(sessionId: String) = flowOf(0)
    override suspend fun upsertSession(session: ShoeSession) { this.session = session }
    override suspend fun appendEvent(event: CardLedgerEvent) { events += event }
}
