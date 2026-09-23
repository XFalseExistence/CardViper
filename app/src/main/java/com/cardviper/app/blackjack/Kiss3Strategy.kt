package com.cardviper.app.blackjack

import com.cardviper.app.model.CardColor
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.PlayingCard

class Kiss3Strategy : CountStrategy {
    override val id: CountStrategyId = CountStrategyId.KISS_III
    override val version: Int = 1

    override fun valueOf(card: PlayingCard): Int? = when (card.rank) {
        CardRank.TWO -> when (card.color) {
            CardColor.BLACK -> 1
            CardColor.RED -> 0
            null -> null
        }

        CardRank.THREE,
        CardRank.FOUR,
        CardRank.FIVE,
        CardRank.SIX,
        CardRank.SEVEN -> 1

        CardRank.EIGHT,
        CardRank.NINE -> 0

        CardRank.TEN,
        CardRank.JACK,
        CardRank.QUEEN,
        CardRank.KING,
        CardRank.ACE -> -1
    }

    override fun initialRunningCount(deckCount: Int): Int {
        require(deckCount == SIX_DECK_PROFILE) {
            "KISS III V1 currently defines only the six-deck profile"
        }
        return 9
    }

    override fun keyCount(deckCount: Int): Int? = if (deckCount == SIX_DECK_PROFILE) 20 else null

    override fun insuranceCount(deckCount: Int): Int? = if (deckCount == SIX_DECK_PROFILE) 25 else null

    private companion object {
        const val SIX_DECK_PROFILE = 6
    }
}
