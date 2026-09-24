package com.cardviper.app.vision

import org.junit.Assert.*
import org.junit.Test

class ImageRecognitionTypesTest {
    private fun observation(confidence: Float) = ImageCardObservation(
        candidate = CardCandidate(0f, 0f, 1f, 1f, 1f),
        cropRect = PixelRect(0, 0, 1, 1),
        identity = CardIdentity.Back,
        classifierConfidence = confidence,
        uncertain = false,
    )

    @Test fun validConstructionAllowsConfidenceEndpointsAndSignedDeltas() {
        listOf(0f, 0.5f, 1f).forEach { confidence ->
            val card = observation(confidence)
            assertEquals(confidence, card.classifierConfidence, 0f)
            assertTrue(card.alternatives.isEmpty())
        }
        val result = ImageRecognitionResult(listOf(observation(1f)), -1, 1, 0, 0)
        assertEquals(-1, result.koDelta)
        assertEquals(1, result.kiss3Delta)
        assertEquals(0, result.uncertainCount)
        assertEquals(0, result.skippedCandidates)
        assertEquals(2, result.copy(uncertainCount = 2, skippedCandidates = 3).uncertainCount)
        val progress = RecognitionProgress(RecognitionStage.DETECTING)
        assertEquals(0, progress.completed)
        assertEquals(0, progress.total)
        assertEquals(2, RecognitionProgress(RecognitionStage.CLASSIFYING, 2, 3).completed)
    }

    @Test fun rejectsNonFiniteAndOutOfRangeClassifierConfidence() {
        listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0.1f, 1.1f)
            .forEach { invalid ->
                assertThrows(IllegalArgumentException::class.java) { observation(invalid) }
            }
    }

    @Test fun rejectsNegativeProgressCounters() {
        assertThrows(IllegalArgumentException::class.java) {
            RecognitionProgress(RecognitionStage.CLASSIFYING, completed = -1)
        }
        assertThrows(IllegalArgumentException::class.java) {
            RecognitionProgress(RecognitionStage.DETECTING, total = -1)
        }
    }

    @Test fun rejectsNegativeResultCounters() {
        assertThrows(IllegalArgumentException::class.java) {
            ImageRecognitionResult(emptyList(), 0, 0, -1, 0)
        }
        assertThrows(IllegalArgumentException::class.java) {
            ImageRecognitionResult(emptyList(), 0, 0, 0, -1)
        }
    }

    @Test fun inferenceErrorsPreserveDiagnosticMessageAndCause() {
        val cause = IllegalStateException("inference failed")
        val detector = DetectorInferenceException("Detector failed", cause)
        val classifier = ClassifierInferenceException("Classifier failed", cause)
        assertEquals("Detector failed", detector.message)
        assertEquals("Classifier failed", classifier.message)
        assertSame(cause, detector.cause)
        assertSame(cause, classifier.cause)
        assertNull(DetectorInferenceException("Detector failed").cause)
        assertNull(ClassifierInferenceException("Classifier failed").cause)
    }
}
