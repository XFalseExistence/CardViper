package com.cardviper.app.vision.litert

import com.cardviper.app.vision.CardIdentity
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.PlayingCard
import org.junit.Assert.*
import org.junit.Test

class LiteRtClassifierPostprocessorTest {
    @Test fun mapsExactly53ScoresIncludingBackAcTenHeartsAndKingSpades() {
        val scores = FloatArray(53)
        scores[0] = 0.3f
        scores[38] = 0.8f
        scores[51] = 0.5f
        scores[52] = 1f
        val prediction = LiteRtClassifierPostprocessor.process(scores, contract(), canonicalLabels())
        assertEquals(CardIdentity.Back, prediction.identity)
        assertEquals("BACK", prediction.label)
        assertEquals(CardIdentity.Face(PlayingCard(CardRank.TEN, CardSuit.HEARTS)), prediction.alternatives[0].identity)
        assertEquals(CardIdentity.Face(PlayingCard(CardRank.KING, CardSuit.SPADES)), prediction.alternatives[1].identity)
        assertEquals(CardIdentity.Face(PlayingCard(CardRank.ACE, CardSuit.CLUBS)), prediction.alternatives[2].identity)
    }

    @Test fun tiesUseLowerCanonicalIndexAndConfidenceBoundariesAreInclusive() {
        val scores = FloatArray(53)
        scores[0] = 0f
        scores[1] = 0f
        val zero = LiteRtClassifierPostprocessor.process(scores, contract(), canonicalLabels())
        assertEquals("AC", zero.label)
        scores[0] = 1f
        val one = LiteRtClassifierPostprocessor.process(scores, contract(), canonicalLabels())
        assertEquals(1f, one.confidence)
    }

    @Test fun rejectsWrongWidthNonfiniteScoresAndLabelOrderMismatch() {
        assertThrows(IllegalArgumentException::class.java) { LiteRtClassifierPostprocessor.process(FloatArray(52), contract(), canonicalLabels()) }
        listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY).forEach { bad ->
            val scores = FloatArray(53)
            scores[2] = bad
            assertThrows(IllegalArgumentException::class.java) { LiteRtClassifierPostprocessor.process(scores, contract(), canonicalLabels()) }
        }
        val wrongLabels = canonicalLabels().toMutableList().also { java.util.Collections.swap(it, 0, 1) }
        assertThrows(IllegalArgumentException::class.java) { LiteRtClassifierPostprocessor.process(FloatArray(53), contract(), wrongLabels) }
    }

    @Test fun rejectsValuesOutsideConfidenceRange() {
        val scores = FloatArray(53)
        scores[10] = 1.01f
        assertThrows(IllegalArgumentException::class.java) { LiteRtClassifierPostprocessor.process(scores, contract(), canonicalLabels()) }
    }

    @Test fun topKIncludesBestAndReturnsOnlyTopKMinusOneAlternatives() {
        val base = contract()
        val oneBest = base.copy(classifier = base.classifier!!.copy(topK = 1))
        val result = LiteRtClassifierPostprocessor.process(FloatArray(53), oneBest, canonicalLabels())
        assertEquals("AC", result.label)
        assertTrue(result.alternatives.isEmpty())
    }

    private fun contract() = LiteRtModelContract.parse(fixture())
    private fun fixture() = javaClass.getResource("/vision/classifier.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")!!.readText()
    private fun canonicalLabels(): List<String> = listOf("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
        .flatMap { rank -> listOf("C", "D", "H", "S").map { "$rank$it" } } + "BACK"
}
