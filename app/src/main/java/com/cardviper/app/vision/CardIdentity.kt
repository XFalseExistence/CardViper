package com.cardviper.app.vision

import com.cardviper.app.model.PlayingCard

/** The 52 exact face identities and one explicit face-down identity. */
sealed interface CardIdentity {
    data class Face(val card: PlayingCard) : CardIdentity {
        init {
            require(card.suit != null) { "A face identity requires an exact suit" }
        }
    }

    data object Back : CardIdentity
}
