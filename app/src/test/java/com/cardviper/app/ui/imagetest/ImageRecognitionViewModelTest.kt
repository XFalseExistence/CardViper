package com.cardviper.app.ui.imagetest

import androidx.lifecycle.ViewModelStore
import com.cardviper.app.vision.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class ImageRecognitionViewModelTest {
    private val dispatcher = StandardTestDispatcher()
    private val store = ViewModelStore()
    private val image = VisionImage(2, 1, ByteArray(6))
    private val observation = ImageCardObservation(CardCandidate(0f, 0f, 1f, 1f, 1f),
        PixelRect(0, 0, 1, 1), CardIdentity.Back, 1f, false)
    private val result = ImageRecognitionResult(listOf(observation), 0, 0, 0, 0)

    @Before fun setup() { Dispatchers.setMain(dispatcher) }
    @After fun tearDown() { store.clear(); Dispatchers.resetMain() }

    private fun model(
        loader: ImageSourceLoader = ImageSourceLoader { image },
        recognize: suspend (VisionImage, (RecognitionProgress) -> Unit) -> ImageRecognitionResult = { _, _ -> result },
    ): ImageRecognitionViewModel {
        val engine = object : ImageRecognitionEngine {
            override suspend fun recognize(image: VisionImage, onProgress: (RecognitionProgress) -> Unit) = recognize(image, onProgress)
        }
        return ImageRecognitionViewModel(loader, engine).also { store.put("model", it) }
    }
    private suspend fun ImageRecognitionViewModel.phase(phase: ImageTestPhase) = uiState.first { it.phase == phase }

    @Test fun progressesFromIdleThroughLoadingAndInferenceToCompleteWithSameImage() = runTest(dispatcher) {
        val loaded = CompletableDeferred<Unit>()
        val classify = CompletableDeferred<Unit>()
        val finish = CompletableDeferred<Unit>()
        val mainThread = Thread.currentThread()
        val model = model(ImageSourceLoader { loaded.await(); image }) { actual, progress ->
            assertSame(image, actual)
            assertNotSame(mainThread, Thread.currentThread())
            progress(RecognitionProgress(RecognitionStage.DETECTING))
            classify.await()
            progress(RecognitionProgress(RecognitionStage.CLASSIFYING, 1, 2))
            finish.await()
            result
        }
        assertEquals(ImageTestPhase.IDLE, model.uiState.value.phase)
        model.analyze("content://test/cards")
        assertEquals(ImageTestPhase.LOADING_IMAGE, model.uiState.value.phase)
        assertNull(model.uiState.value.image)
        loaded.complete(Unit)
        assertSame(image, model.phase(ImageTestPhase.DETECTING).image)
        classify.complete(Unit)
        val classifying = model.phase(ImageTestPhase.CLASSIFYING)
        assertEquals("CLASSIFYING 1/2", classifying.statusText)
        assertEquals(RecognitionProgress(RecognitionStage.CLASSIFYING, 1, 2), classifying.progress)
        finish.complete(Unit)
        val complete = model.phase(ImageTestPhase.COMPLETE)
        assertEquals("content://test/cards", complete.source)
        assertSame(image, complete.image)
        assertSame(result, complete.result)
    }

    @Test fun noCardsIsARecoverableNonErrorState() = runTest(dispatcher) {
        var calls = 0
        val model = model { _, _ -> if (calls++ == 0) ImageRecognitionResult(emptyList(), 0, 0, 0, 0) else result }
        model.analyze("content://test/empty")
        val empty = model.phase(ImageTestPhase.NO_CARDS)
        assertEquals("NO CARDS DETECTED", empty.statusText)
        assertSame(image, empty.image)
        assertTrue(empty.result!!.observations.isEmpty())
        model.analyze("content://test/cards")
        assertSame(result, model.phase(ImageTestPhase.COMPLETE).result)
    }

    @Test fun typedErrorsHaveDistinctStableCopyAndEachCanRecover() = runTest(dispatcher) {
        val failures = listOf(
            ImageLoadException("private uri details") to "COULD NOT OPEN IMAGE",
            ImageDecodeException("decoder details") to "COULD NOT DECODE IMAGE",
            VisionModelUnavailableException("model details") to "VISION MODEL NOT INSTALLED",
            DetectorInferenceException("detector details") to "CARD DETECTOR FAILED",
            ClassifierInferenceException("classifier details") to "CARD CLASSIFIER FAILED",
        )
        for ((failure, expected) in failures) {
            var fail = true
            val isLoading = failure is ImageLoadException || failure is ImageDecodeException
            val model = model(ImageSourceLoader { if (fail && isLoading) throw failure; image }) { _, _ ->
                if (fail) throw failure
                result
            }
            model.analyze("content://test/failure")
            val error = model.phase(ImageTestPhase.ERROR)
            assertEquals(expected, error.statusText)
            assertNull(error.result)
            if (isLoading) assertNull(error.image) else assertSame(image, error.image)
            fail = false
            model.analyze("content://test/recovered")
            assertEquals("content://test/recovered", model.phase(ImageTestPhase.COMPLETE).source)
        }
    }

    @Test fun newestSourceCancelsPreviousLoading() = runTest(dispatcher) {
        val started = CompletableDeferred<Unit>()
        val cancelled = CompletableDeferred<Unit>()
        val model = model(ImageSourceLoader { source ->
            if (source.endsWith("old")) {
                started.complete(Unit)
                try { awaitCancellation() } finally { cancelled.complete(Unit) }
            }
            image
        })
        model.analyze("content://test/old")
        started.await()
        model.analyze("content://test/new")
        cancelled.await()
        assertEquals("content://test/new", model.phase(ImageTestPhase.COMPLETE).source)
    }

    @Test fun cancelledInferenceCannotOverwriteNewerAnalysisEvenForSameUri() = runTest(dispatcher) {
        val firstImage = VisionImage(1, 1, ByteArray(3))
        var loads = 0
        val oldStarted = CompletableDeferred<Unit>()
        val cancelled = CompletableDeferred<Unit>()
        val releaseOld = CompletableDeferred<Unit>()
        val oldFinished = CompletableDeferred<Unit>()
        val model = model(ImageSourceLoader { if (loads++ == 0) firstImage else image }) { decoded, progress ->
            if (decoded === firstImage) {
                oldStarted.complete(Unit)
                try { awaitCancellation() } finally {
                    cancelled.complete(Unit)
                    withContext(NonCancellable) {
                        releaseOld.await()
                        progress(RecognitionProgress(RecognitionStage.CLASSIFYING, 99, 100))
                        oldFinished.complete(Unit)
                    }
                }
            }
            result
        }
        model.analyze("content://test/same")
        oldStarted.await()
        model.analyze("content://test/same")
        cancelled.await()
        assertSame(image, model.phase(ImageTestPhase.COMPLETE).image)
        releaseOld.complete(Unit)
        oldFinished.await()
        runCurrent()
        assertEquals(ImageTestPhase.COMPLETE, model.uiState.value.phase)
        assertSame(image, model.uiState.value.image)
        assertSame(result, model.uiState.value.result)
    }

    @Test fun resetCancelsWorkClearsImageAndAllowsAnotherAnalysis() = runTest(dispatcher) {
        val started = CompletableDeferred<Unit>()
        val cancelled = CompletableDeferred<Unit>()
        var block = true
        val model = model { _, _ ->
            if (block) {
                started.complete(Unit)
                try { awaitCancellation() } finally { cancelled.complete(Unit) }
            }
            result
        }
        model.analyze("content://test/old")
        started.await()
        model.reset()
        cancelled.await()
        assertEquals(ImageTestPhase.IDLE, model.uiState.value.phase)
        assertNull(model.uiState.value.source)
        assertNull(model.uiState.value.image)
        assertNull(model.uiState.value.result)
        block = false
        model.analyze("content://test/new")
        assertSame(result, model.phase(ImageTestPhase.COMPLETE).result)
    }

    @Test fun unknownFailureIsRecoverableWithoutLeakingExceptionDetails() = runTest(dispatcher) {
        var fail = true
        val model = model { _, _ -> if (fail) error("sensitive implementation detail"); result }
        model.analyze("content://test/error")
        assertEquals("IMAGE ANALYSIS FAILED", model.phase(ImageTestPhase.ERROR).statusText)
        fail = false
        model.analyze("content://test/new")
        model.phase(ImageTestPhase.COMPLETE)
    }

    @Test fun clearingViewModelCancelsInference() = runTest(dispatcher) {
        val started = CompletableDeferred<Unit>()
        val cancelled = CompletableDeferred<Unit>()
        val model = model { _, _ ->
            started.complete(Unit)
            try { awaitCancellation() } finally { cancelled.complete(Unit) }
        }
        model.analyze("content://test/cards")
        started.await()
        store.clear()
        cancelled.await()
        assertNotEquals(ImageTestPhase.ERROR, model.uiState.value.phase)
    }
}
