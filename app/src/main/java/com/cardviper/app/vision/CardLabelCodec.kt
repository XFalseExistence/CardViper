package com.cardviper.app.vision

import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.PlayingCard

/** Canonical, case-sensitive ASCII labels. This does not define model tensor ordering. */
object CardLabelCodec {
    private val rankToToken = mapOf(
        CardRank.ACE to "A", CardRank.TWO to "2", CardRank.THREE to "3",
        CardRank.FOUR to "4", CardRank.FIVE to "5", CardRank.SIX to "6",
        CardRank.SEVEN to "7", CardRank.EIGHT to "8", CardRank.NINE to "9",
        CardRank.TEN to "10", CardRank.JACK to "J", CardRank.QUEEN to "Q",
        CardRank.KING to "K",
    )
    private val suitToToken = mapOf(
        CardSuit.CLUBS to "C", CardSuit.DIAMONDS to "D",
        CardSuit.HEARTS to "H", CardSuit.SPADES to "S",
    )
    private val tokenToRank = rankToToken.entries.associate { (rank, token) -> token to rank }
    private val tokenToSuit = suitToToken.entries.associate { (suit, token) -> token to suit }

    fun decode(label: String): CardIdentity {
        if (label == "BACK") return CardIdentity.Back
        require(label.length in 2..3) { "Invalid card label: $label" }
        val rank = requireNotNull(tokenToRank[label.dropLast(1)]) { "Invalid card rank label: $label" }
        val suit = requireNotNull(tokenToSuit[label.takeLast(1)]) { "Invalid card suit label: $label" }
        return CardIdentity.Face(PlayingCard(rank, suit))
    }

    fun encode(identity: CardIdentity): String = when (identity) {
        CardIdentity.Back -> "BACK"
        is CardIdentity.Face -> rankToToken.getValue(identity.card.rank) +
            suitToToken.getValue(requireNotNull(identity.card.suit))
    }
}
