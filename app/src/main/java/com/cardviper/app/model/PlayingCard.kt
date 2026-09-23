package com.cardviper.app.model

data class PlayingCard(
    val rank: CardRank,
    val suit: CardSuit? = null,
    val colorHint: CardColor? = null,
) {
    val color: CardColor?
        get() = suit?.color ?: colorHint

    init {
        require(suit == null || colorHint == null || suit.color == colorHint) {
            "colorHint must agree with suit"
        }
    }
}
