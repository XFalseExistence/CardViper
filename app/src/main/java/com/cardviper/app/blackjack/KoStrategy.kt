package com.cardviper.app.blackjack

import com.cardviper.app.model.CardRank
import com.cardviper.app.model.PlayingCard

class KoStrategy : CountStrategy {
    override val id: CountStrategyId = CountStrategyId.KO
    override val version: Int = 1

    override fun valueOf(card: PlayingCard): Int = when (card.rank) {
        CardRank.TWO,
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
        require(deckCount > 0) { "deckCount must be positive" }
        return 4 - (4 * deckCount)
    }
}
