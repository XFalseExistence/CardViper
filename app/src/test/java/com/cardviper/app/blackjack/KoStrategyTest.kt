package com.cardviper.app.blackjack

import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.PlayingCard
import org.junit.Assert.assertEquals
import org.junit.Test

class KoStrategyTest {
    private val strategy = KoStrategy()

    @Test
    fun everyRankHasClassicKoTag() {
        val expected = mapOf(
            CardRank.TWO to 1,
            CardRank.THREE to 1,
            CardRank.FOUR to 1,
            CardRank.FIVE to 1,
            CardRank.SIX to 1,
            CardRank.SEVEN to 1,
            CardRank.EIGHT to 0,
            CardRank.NINE to 0,
            CardRank.TEN to -1,
            CardRank.JACK to -1,
            CardRank.QUEEN to -1,
            CardRank.KING to -1,
            CardRank.ACE to -1,
        )
        expected.forEach { (rank, tag) ->
            assertEquals(tag, strategy.valueOf(PlayingCard(rank, CardSuit.CLUBS)))
        }
    }

    @Test
    fun ircFollowsClassicKoFormula() {
        assertEquals(0, strategy.initialRunningCount(1))
        assertEquals(-4, strategy.initialRunningCount(2))
        assertEquals(-12, strategy.initialRunningCount(4))
        assertEquals(-20, strategy.initialRunningCount(6))
        assertEquals(-28, strategy.initialRunningCount(8))
    }

    @Test
    fun knownSequenceHasNetDeltaPlusOne() {
        val cards = listOf(
            PlayingCard(CardRank.TWO),
            PlayingCard(CardRank.FIVE),
            PlayingCard(CardRank.KING),
            PlayingCard(CardRank.EIGHT),
            PlayingCard(CardRank.ACE),
            PlayingCard(CardRank.SEVEN),
        )
        val snapshot = strategy.evaluate(0, cards)
        assertEquals(1, snapshot.netDelta)
        assertEquals(1, snapshot.runningCount)
    }

    @Test
    fun fullDeckNetIsPlusFourAndSixDecksScaleLinearly() {
        val deck = CardSuit.entries.flatMap { suit ->
            CardRank.entries.map { rank -> PlayingCard(rank, suit) }
        }
        assertEquals(4, strategy.evaluate(0, deck).netDelta)
        assertEquals(24, strategy.evaluate(0, List(6) { deck }.flatten()).netDelta)
        assertEquals(4, strategy.evaluate(strategy.initialRunningCount(6), List(6) { deck }.flatten()).runningCount)
    }
}
