package com.cardviper.app.blackjack

import com.cardviper.app.model.CardEventSource
import com.cardviper.app.model.CardLedgerEvent
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.LedgerEventType
import com.cardviper.app.model.PlayingCard
import org.junit.Assert.assertEquals
import org.junit.Test

class LedgerResolverTest {
    private val resolver = LedgerResolver()

    @Test
    fun correctionInvalidationAndManualAddResolveToEffectiveSequence() {
        val events = listOf(
            event("001", 1, LedgerEventType.CARD_COMMITTED, card = card(CardRank.KING, CardSuit.HEARTS)),
            event("002", 2, LedgerEventType.CARD_COMMITTED, card = card(CardRank.SIX, CardSuit.CLUBS)),
            event("003", 3, LedgerEventType.CARD_CORRECTED, target = "001", card = card(CardRank.QUEEN, CardSuit.HEARTS)),
            event("004", 4, LedgerEventType.CARD_INVALIDATED, target = "002"),
            event("005", 5, LedgerEventType.CARD_MANUAL_ADDED, card = card(CardRank.THREE, CardSuit.DIAMONDS), source = CardEventSource.MANUAL),
        )

        assertEquals(
            listOf(card(CardRank.QUEEN, CardSuit.HEARTS), card(CardRank.THREE, CardSuit.DIAMONDS)),
            resolver.resolve(events).map { it.card },
        )
    }

    @Test
    fun latestCorrectionWinsEvenWhenItTargetsEarlierCorrection() {
        val events = listOf(
            event("001", 1, LedgerEventType.CARD_COMMITTED, card = card(CardRank.KING, CardSuit.HEARTS)),
            event("002", 2, LedgerEventType.CARD_CORRECTED, target = "001", card = card(CardRank.QUEEN, CardSuit.HEARTS)),
            event("003", 3, LedgerEventType.CARD_CORRECTED, target = "002", card = card(CardRank.FIVE, CardSuit.HEARTS)),
        )

        assertEquals(listOf(card(CardRank.FIVE, CardSuit.HEARTS)), resolver.resolve(events).map { it.card })
    }

    @Test
    fun invalidatingAnyEventInACorrectionChainInvalidatesThePhysicalCard() {
        val events = listOf(
            event("001", 1, LedgerEventType.CARD_COMMITTED, card = card(CardRank.KING, CardSuit.HEARTS)),
            event("002", 2, LedgerEventType.CARD_CORRECTED, target = "001", card = card(CardRank.QUEEN, CardSuit.HEARTS)),
            event("003", 3, LedgerEventType.CARD_INVALIDATED, target = "002"),
        )

        assertEquals(emptyList<ResolvedCard>(), resolver.resolve(events))
    }

    @Test
    fun unknownTargetsAreIgnoredDeterministically() {
        val events = listOf(
            event("001", 1, LedgerEventType.CARD_COMMITTED, card = card(CardRank.SEVEN, CardSuit.SPADES)),
            event("002", 2, LedgerEventType.CARD_CORRECTED, target = "missing", card = card(CardRank.ACE, CardSuit.SPADES)),
            event("003", 3, LedgerEventType.CARD_INVALIDATED, target = "also-missing"),
        )

        assertEquals(listOf(card(CardRank.SEVEN, CardSuit.SPADES)), resolver.resolve(events).map { it.card })
    }

    private fun card(rank: CardRank, suit: CardSuit) = PlayingCard(rank, suit)

    private fun event(
        id: String,
        sequence: Long,
        type: LedgerEventType,
        target: String? = null,
        card: PlayingCard? = null,
        source: CardEventSource = CardEventSource.VISION,
    ) = CardLedgerEvent(
        eventId = id,
        sessionId = "session-1",
        sequenceNumber = sequence,
        timestampEpochMillis = sequence,
        eventType = type,
        targetEventId = target,
        trackId = null,
        card = card,
        source = source,
    )
}
