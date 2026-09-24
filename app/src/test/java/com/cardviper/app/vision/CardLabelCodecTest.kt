package com.cardviper.app.vision

import com.cardviper.app.model.CardColor
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.PlayingCard
import org.junit.Assert.*
import org.junit.Test

class CardLabelCodecTest {
    @Test fun all52FacesAndBackRoundTrip() {
        val faces = CardRank.entries.flatMap { rank ->
            CardSuit.entries.map { suit -> CardIdentity.Face(PlayingCard(rank, suit)) }
        }
        assertEquals(52, faces.size)
        val identities = faces + CardIdentity.Back
        assertEquals(53, identities.map(CardLabelCodec::encode).toSet().size)
        identities.forEach { identity ->
            assertEquals(identity, CardLabelCodec.decode(CardLabelCodec.encode(identity)))
        }
        assertEquals(CardIdentity.Back, CardLabelCodec.decode("BACK"))
        assertEquals("BACK", CardLabelCodec.encode(CardIdentity.Back))
    }

    @Test fun canonicalLabelsRepresentTheExpectedRanksAndSuits() {
        val ranks = listOf(
            "A" to CardRank.ACE, "2" to CardRank.TWO, "3" to CardRank.THREE,
            "4" to CardRank.FOUR, "5" to CardRank.FIVE, "6" to CardRank.SIX,
            "7" to CardRank.SEVEN, "8" to CardRank.EIGHT, "9" to CardRank.NINE,
            "10" to CardRank.TEN, "J" to CardRank.JACK, "Q" to CardRank.QUEEN,
            "K" to CardRank.KING,
        )
        val suits = listOf("C" to CardSuit.CLUBS, "D" to CardSuit.DIAMONDS,
            "H" to CardSuit.HEARTS, "S" to CardSuit.SPADES)
        for ((rankToken, rank) in ranks) for ((suitToken, suit) in suits) {
            val label = rankToken + suitToken
            val identity = CardIdentity.Face(PlayingCard(rank, suit))
            assertEquals(label, CardLabelCodec.encode(identity))
            assertEquals(identity, CardLabelCodec.decode(label))
        }
    }

    @Test fun redAndBlackTwosPreserveSuitAndColor() {
        val red = (CardLabelCodec.decode("2H") as CardIdentity.Face).card
        val black = (CardLabelCodec.decode("2S") as CardIdentity.Face).card
        assertEquals(CardRank.TWO, red.rank)
        assertEquals(CardRank.TWO, black.rank)
        assertEquals(CardSuit.HEARTS, red.suit)
        assertEquals(CardSuit.SPADES, black.suit)
        assertEquals(CardColor.RED, red.color)
        assertEquals(CardColor.BLACK, black.color)
    }

    @Test fun malformedLabelsAreRejectedRatherThanGuessed() {
        listOf("", "A", "S", "1S", "11H", "TS", "0C", "BACKS", "back", "2h",
            " AS", "AS ", "AS\n", "7♠", "2HS", "AS\u0000").forEach { label ->
            assertThrows("label=$label", IllegalArgumentException::class.java) {
                CardLabelCodec.decode(label)
            }
        }
    }

    @Test fun faceIdentityRequiresAnExactSuit() {
        assertThrows(IllegalArgumentException::class.java) {
            CardIdentity.Face(PlayingCard(CardRank.TWO))
        }
        assertThrows(IllegalArgumentException::class.java) {
            CardIdentity.Face(PlayingCard(CardRank.TWO, colorHint = CardColor.BLACK))
        }
    }
}
