package com.cardviper.app.vision

import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Test

class VisionContractsTest {
    @Test fun noOpDetectorIsAnEmptyDeterministicFake() = runTest {
        val detector: CardDetector = NoOpCardDetector()
        val image = VisionImage(2, 1, ByteArray(6))
        assertTrue(detector.detect(image).isEmpty())
        assertTrue(detector.detect(image).isEmpty())
    }

    @Test fun unavailableDetectorReportsMissingModelInsteadOfNoCards() = runTest {
        val detector: CardDetector = UnavailableCardDetector()
        val failure = runCatching { detector.detect(VisionImage(1, 1, ByteArray(3))) }.exceptionOrNull()
        assertTrue(failure is VisionModelUnavailableException)
        assertEquals("Card detector model is not installed", failure?.message)
    }

    @Test fun noOpRecognizerReportsMissingModelInsteadOfGuessingOrReturningNull() = runTest {
        val recognizer: CardRecognizer = NoOpCardRecognizer()
        val failure = runCatching { recognizer.recognize(VisionImage(1, 1, ByteArray(3))) }.exceptionOrNull()
        assertTrue(failure is VisionModelUnavailableException)
        assertEquals("Card classifier is not configured", failure?.message)
    }

    @Test fun candidateCoordinatesAreFiniteSourcePixelsAndMayNeedClampingOrSkipping() {
        val sourcePixels = CardCandidate(120.5f, -4f, 30f, 18f, 1f)
        assertEquals(120.5f, sourcePixels.x, 0f)
        assertEquals(-4f, sourcePixels.y, 0f)
        // The next task's crop engine must receive degenerate detections to skip safely.
        assertEquals(0f, CardCandidate(0f, 0f, 0f, -2f, 0f).width, 0f)
        listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY).forEach { invalid ->
            assertThrows(IllegalArgumentException::class.java) { CardCandidate(invalid, 0f, 1f, 1f, 1f) }
            assertThrows(IllegalArgumentException::class.java) { CardCandidate(0f, invalid, 1f, 1f, 1f) }
            assertThrows(IllegalArgumentException::class.java) { CardCandidate(0f, 0f, invalid, 1f, 1f) }
            assertThrows(IllegalArgumentException::class.java) { CardCandidate(0f, 0f, 1f, invalid, 1f) }
        }
    }

    @Test fun confidenceCannotBeNonFiniteOrOutsideProbabilityRange() {
        listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0.1f, 1.1f).forEach { invalid ->
            assertThrows(IllegalArgumentException::class.java) { CardCandidate(0f, 0f, 1f, 1f, invalid) }
            assertThrows(IllegalArgumentException::class.java) { CardRecognition(CardIdentity.Back, invalid) }
            assertThrows(IllegalArgumentException::class.java) { RankedIdentity(CardIdentity.Back, invalid) }
        }
        assertTrue(CardRecognition(CardIdentity.Back, 0f).alternatives.isEmpty())
        assertEquals(1f, RankedIdentity(CardIdentity.Back, 1f).confidence, 0f)
    }
}
