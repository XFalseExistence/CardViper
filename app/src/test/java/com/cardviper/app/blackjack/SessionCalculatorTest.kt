package com.cardviper.app.blackjack

import com.cardviper.app.model.CardEventSource
import com.cardviper.app.model.CardLedgerEvent
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.LedgerEventType
import com.cardviper.app.model.PlayingCard
import org.junit.Assert.assertEquals
import org.junit.Test

class SessionCalculatorTest {
    private val resolver = LedgerResolver()
    private val calculator = SessionCalculator()

    @Test
    fun correctionReplaysFromFactsInsteadOfIncrementallyPatchingCount() {
        val initial = listOf(
            commit("001", 1, PlayingCard(CardRank.KING, CardSuit.HEARTS)),
            commit("002", 2, PlayingCard(CardRank.EIGHT, CardSuit.SPADES)),
        )
        val corrected = initial + CardLedgerEvent(
            eventId = "003",
            sessionId = "session-1",
            sequenceNumber = 3,
            timestampEpochMillis = 3,
            eventType = LedgerEventType.CARD_CORRECTED,
            targetEventId = "001",
            card = PlayingCard(CardRank.FIVE, CardSuit.HEARTS),
            source = CardEventSource.CORRECTION,
        )
        val strategy = KoStrategy()
        val starting = strategy.initialRunningCount(6)

        val snapshot = calculator.calculate(starting, strategy, resolver.resolve(corrected))
        val freshReplay = strategy.evaluate(starting, listOf(PlayingCard(CardRank.FIVE, CardSuit.HEARTS), PlayingCard(CardRank.EIGHT, CardSuit.SPADES)))

        assertEquals(freshReplay, snapshot)
        assertEquals(-19, snapshot.runningCount)
    }

    @Test
    fun switchingKoToKissAndBackDoesNotMutateLedger() {
        val events = listOf(
            commit("001", 1, PlayingCard(CardRank.TWO, CardSuit.SPADES)),
            commit("002", 2, PlayingCard(CardRank.KING, CardSuit.HEARTS)),
            commit("003", 3, PlayingCard(CardRank.SEVEN, CardSuit.CLUBS)),
        )
        val before = events.toList()
        val resolved = resolver.resolve(events)
        val ko = KoStrategy()
        val kiss = Kiss3Strategy()

        val firstKo = calculator.calculate(ko.initialRunningCount(6), ko, resolved)
        calculator.calculate(kiss.initialRunningCount(6), kiss, resolved)
        val secondKo = calculator.calculate(ko.initialRunningCount(6), ko, resolver.resolve(events))

        assertEquals(before, events)
        assertEquals(firstKo, secondKo)
    }

    private fun commit(id: String, sequence: Long, card: PlayingCard) = CardLedgerEvent(
        eventId = id,
        sessionId = "session-1",
        sequenceNumber = sequence,
        timestampEpochMillis = sequence,
        eventType = LedgerEventType.CARD_COMMITTED,
        card = card,
        source = CardEventSource.VISION,
    )
}
