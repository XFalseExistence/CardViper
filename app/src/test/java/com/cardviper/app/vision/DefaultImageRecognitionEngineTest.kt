package com.cardviper.app.vision

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Test

class DefaultImageRecognitionEngineTest {
    private val box = CardCandidate(20f, 20f, 10f, 10f, 0.9f)
    private fun image() = VisionImage(100, 80, ByteArray(100 * 80 * 3) { (it % 127).toByte() })
    private fun prediction(label: String, confidence: Float = 1f) =
        CardRecognition(CardLabelCodec.decode(label), confidence)

    private class FakeDetector(val action: suspend (VisionImage) -> List<CardCandidate>) : CardDetector {
        override suspend fun detect(image: VisionImage) = action(image)
    }
    private class FakeRecognizer(val action: suspend (VisionImage) -> CardRecognition) : CardRecognizer {
        override suspend fun recognize(crop: VisionImage) = action(crop)
    }
    private fun engine(
        boxes: List<CardCandidate> = listOf(box),
        prediction: CardRecognition = prediction("BACK"),
        threshold: Float = 0.8f,
        padding: Float = 0.1f,
        classify: suspend (VisionImage) -> CardRecognition = { prediction },
    ) = DefaultImageRecognitionEngine(FakeDetector { boxes }, FakeRecognizer(classify), threshold, padding)

    @Test fun backIsVisibleButCountsZero() = runTest {
        val result = engine().recognize(image())
        assertEquals(CardIdentity.Back, result.observations.single().identity)
        assertEquals(0, result.koDelta)
        assertEquals(0, result.kiss3Delta)
        assertEquals(0, result.uncertainCount)
    }

    @Test fun lowConfidenceFaceRemainsInPreviewWithAlternatives() = runTest {
        val alternatives = listOf(RankedIdentity(CardLabelCodec.decode("6C"), 0.3f))
        val output = prediction("5C", 0.55f).copy(alternatives = alternatives)
        val result = engine(prediction = output).recognize(image())
        val card = result.observations.single()
        assertEquals(CardLabelCodec.decode("5C"), card.identity)
        assertEquals(0.55f, card.classifierConfidence, 0f)
        assertEquals(alternatives, card.alternatives)
        assertTrue(card.uncertain)
        assertEquals(1, result.uncertainCount)
        assertEquals(1, result.koDelta)
        assertEquals(1, result.kiss3Delta)
    }

    @Test fun confidenceThresholdIsInclusiveAndConfigurableForFacesAndBacks() = runTest {
        for (label in listOf("5C", "BACK")) {
            assertFalse(engine(prediction = prediction(label, 0.8f)).recognize(image()).observations.single().uncertain)
            assertTrue(engine(prediction = prediction(label, 0.8f), threshold = 0.9f)
                .recognize(image()).observations.single().uncertain)
        }
    }

    @Test fun kissDifferentiatesBothRedAndBothBlackTwos() = runTest {
        for ((label, kiss) in listOf("2H" to 0, "2D" to 0, "2S" to 1, "2C" to 1)) {
            val result = engine(prediction = prediction(label)).recognize(image())
            assertEquals(label, 1, result.koDelta)
            assertEquals(label, kiss, result.kiss3Delta)
        }
    }

    @Test fun mixedImageDeltasAreLocalAndDoNotAccumulateBetweenCalls() = runTest {
        val labels = listOf("2H", "2S", "7C", "9D", "KS", "AC", "BACK")
        var calls = 0
        val engine = engine(boxes = labels.mapIndexed { index, _ -> box.copy(x = index * 10f) }) {
            prediction(labels[calls++ % labels.size])
        }
        repeat(2) {
            val result = engine.recognize(image())
            assertEquals(1, result.koDelta)
            assertEquals(0, result.kiss3Delta)
            assertEquals(7, result.observations.size)
        }
        assertEquals(14, calls)
    }

    @Test fun sortsByTopThenLeftThenConfidenceAndPreservesCandidateCoordinates() = runTest {
        val first = box.copy(x = 5f, y = 1f)
        val second = box.copy(x = 10f, y = 1f, confidence = 0.9f)
        val third = second.copy(confidence = 0.2f)
        val fourth = box.copy(x = 0f, y = 2f)
        var calls = 0
        val engine = engine(boxes = listOf(fourth, third, first, second)) {
            prediction(listOf("AC", "2H", "7S", "BACK")[calls++])
        }
        val result = engine.recognize(image())
        assertEquals(listOf(first, second, third, fourth), result.observations.map { it.candidate })
        assertEquals(listOf("AC", "2H", "7S", "BACK"), result.observations.map { CardLabelCodec.encode(it.identity) })
        assertEquals(4, calls)
    }

    @Test fun paddedCropsClampAtAllFourEdges() = runTest {
        val cases = listOf(
            CardCandidate(-5f, 20f, 20f, 20f, 1f) to PixelRect(0, 18, 17, 42),
            CardCandidate(90f, 20f, 20f, 20f, 1f) to PixelRect(88, 18, 100, 42),
            CardCandidate(20f, -5f, 20f, 20f, 1f) to PixelRect(18, 0, 42, 17),
            CardCandidate(20f, 70f, 20f, 20f, 1f) to PixelRect(18, 68, 42, 80),
        )
        for ((candidate, expected) in cases) {
            var received: VisionImage? = null
            val result = engine(boxes = listOf(candidate)) { crop -> received = crop; prediction("BACK") }
                .recognize(image())
            assertEquals(expected, result.observations.single().cropRect)
            val crop = requireNotNull(received)
            assertEquals(expected.right - expected.left, crop.width)
            assertEquals(expected.bottom - expected.top, crop.height)
        }
    }

    @Test fun cropsOriginalSourcePixelsWithoutResizingOrMutatingThem() = runTest {
        val source = VisionImage(4, 3, ByteArray(36) { it.toByte() })
        val detector = FakeDetector { input ->
            assertSame(source, input)
            listOf(CardCandidate(1f, 1f, 2f, 1f, 1f))
        }
        val recognizer = FakeRecognizer { crop ->
            assertEquals(2, crop.width)
            assertEquals(1, crop.height)
            assertArrayEquals(byteArrayOf(15, 16, 17, 18, 19, 20), crop.rgb)
            crop.rgb[0] = 99
            prediction("BACK")
        }
        DefaultImageRecognitionEngine(detector, recognizer, cropPaddingFraction = 0f).recognize(source)
        assertEquals(15.toByte(), source.rgb[15])
    }

    @Test fun fractionalBoundsRoundOutwardAndExtremeFiniteCoordinatesStaySafe() = runTest {
        val fractional = CardCandidate(1.2f, 2.2f, 2.3f, 3.3f, 1f)
        val result = engine(boxes = listOf(fractional), padding = 0f).recognize(image())
        assertEquals(PixelRect(1, 2, 4, 6), result.observations.single().cropRect)
        val huge = CardCandidate(0f, 0f, Float.MAX_VALUE, Float.MAX_VALUE, 1f)
        assertEquals(PixelRect(0, 0, 100, 80), engine(boxes = listOf(huge)).recognize(image()).observations.single().cropRect)
    }

    @Test fun skipsDegenerateAndEntirelyOutsideCandidatesWithoutClassifyingThem() = runTest {
        val invalid = listOf(box.copy(width = 0f), box.copy(height = -1f),
            box.copy(width = -1f), box.copy(height = 0f), box.copy(x = 200f), box.copy(y = -100f))
        var calls = 0
        val result = engine(boxes = invalid + box) { calls++; prediction("BACK") }.recognize(image())
        assertEquals(6, result.skippedCandidates)
        assertEquals(1, result.observations.size)
        assertEquals(1, calls)
    }

    @Test fun noCandidatesAndAllSkippedProduceEmptyZeroPreview() = runTest {
        for (boxes in listOf(emptyList(), listOf(box.copy(width = 0f)))) {
            val result = engine(boxes = boxes) { error("Must not classify invalid or absent candidates") }.recognize(image())
            assertTrue(result.observations.isEmpty())
            assertEquals(0, result.koDelta)
            assertEquals(0, result.kiss3Delta)
            assertEquals(0, result.uncertainCount)
            assertEquals(boxes.size, result.skippedCandidates)
        }
    }

    @Test fun progressCountsOnlyValidCropsAndFinishesAtTotal() = runTest {
        val progress = mutableListOf<RecognitionProgress>()
        engine(boxes = listOf(box, box.copy(x = 40f), box.copy(width = 0f))) {
            assertEquals(RecognitionStage.CLASSIFYING, progress.last().stage)
            prediction("BACK")
        }.recognize(image(), progress::add)
        assertEquals(listOf(
            RecognitionProgress(RecognitionStage.DETECTING),
            RecognitionProgress(RecognitionStage.CLASSIFYING, 0, 2),
            RecognitionProgress(RecognitionStage.CLASSIFYING, 1, 2),
            RecognitionProgress(RecognitionStage.CLASSIFYING, 2, 2),
        ), progress)
    }

    @Test fun wrapsDetectorAndClassifierFailuresWithOriginalCause() = runTest {
        val cause = IllegalStateException("model inference failed")
        val detectorEngine = DefaultImageRecognitionEngine(FakeDetector { throw cause }, FakeRecognizer { error("Must not classify") })
        val detectorFailure = runCatching { detectorEngine.recognize(image()) }.exceptionOrNull()
        assertTrue(detectorFailure is DetectorInferenceException)
        assertSame(cause, detectorFailure!!.cause)
        val classifierFailure = runCatching { engine { throw cause }.recognize(image()) }.exceptionOrNull()
        assertTrue(classifierFailure is ClassifierInferenceException)
        assertSame(cause, classifierFailure!!.cause)
    }

    @Test fun preservesModelUnavailableAndCancellationFromEitherStage() = runTest {
        for (failure in listOf(VisionModelUnavailableException("missing model"), CancellationException("cancelled"))) {
            val detectorEngine = DefaultImageRecognitionEngine(FakeDetector { throw failure }, FakeRecognizer { error("Must not classify") })
            assertSame(failure, runCatching { detectorEngine.recognize(image()) }.exceptionOrNull())
            assertSame(failure, runCatching { engine { throw failure }.recognize(image()) }.exceptionOrNull())
        }
    }

    @Test fun progressCallbackFailuresAreNotMisreportedAsInferenceFailures() = runTest {
        val failure = IllegalStateException("observer failed")
        for (stage in RecognitionStage.entries) {
            assertSame(failure, runCatching {
                engine().recognize(image()) { if (it.stage == stage) throw failure }
            }.exceptionOrNull())
        }
    }

    @Test fun rejectsInvalidThresholdsAndPaddingFractions() {
        for (invalid in listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0.1f, 1.1f)) {
            assertThrows(IllegalArgumentException::class.java) { engine(threshold = invalid) }
            assertThrows(IllegalArgumentException::class.java) { engine(padding = invalid) }
        }
    }
}
