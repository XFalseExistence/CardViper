package com.cardviper.app.blackjack

import com.cardviper.app.model.CardColor
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.PlayingCard
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class Kiss3StrategyTest {
    private val strategy = Kiss3Strategy()

    @Test
    fun twosAreColorAwareAndUnknownColorIsHeld() {
        assertEquals(1, strategy.valueOf(PlayingCard(CardRank.TWO, CardSuit.SPADES)))
        assertEquals(1, strategy.valueOf(PlayingCard(CardRank.TWO, CardSuit.CLUBS)))
        assertEquals(0, strategy.valueOf(PlayingCard(CardRank.TWO, CardSuit.HEARTS)))
        assertEquals(0, strategy.valueOf(PlayingCard(CardRank.TWO, CardSuit.DIAMONDS)))
        assertEquals(1, strategy.valueOf(PlayingCard(CardRank.TWO, colorHint = CardColor.BLACK)))
        assertEquals(0, strategy.valueOf(PlayingCard(CardRank.TWO, colorHint = CardColor.RED)))
        assertNull(strategy.valueOf(PlayingCard(CardRank.TWO)))
    }

    @Test
    fun nonTwoRanksHaveKissThreeTags() {
        listOf(CardRank.THREE, CardRank.FOUR, CardRank.FIVE, CardRank.SIX, CardRank.SEVEN)
            .forEach { assertEquals(1, strategy.valueOf(PlayingCard(it))) }
        listOf(CardRank.EIGHT, CardRank.NINE)
            .forEach { assertEquals(0, strategy.valueOf(PlayingCard(it))) }
        listOf(CardRank.TEN, CardRank.JACK, CardRank.QUEEN, CardRank.KING, CardRank.ACE)
            .forEach { assertEquals(-1, strategy.valueOf(PlayingCard(it))) }
    }

    @Test
    fun sixDeckProfileExposesIrcKeyAndInsurance() {
        assertEquals(9, strategy.initialRunningCount(6))
        assertEquals(20, strategy.keyCount(6))
        assertEquals(25, strategy.insuranceCount(6))
    }

    @Test
    fun fullDeckNetIsPlusTwoAndSixDecksScaleLinearly() {
        val deck = CardSuit.entries.flatMap { suit ->
            CardRank.entries.map { rank -> PlayingCard(rank, suit) }
        }
        assertEquals(2, strategy.evaluate(0, deck).netDelta)
        assertEquals(12, strategy.evaluate(0, List(6) { deck }.flatten()).netDelta)
        assertEquals(21, strategy.evaluate(strategy.initialRunningCount(6), List(6) { deck }.flatten()).runningCount)
    }
}
