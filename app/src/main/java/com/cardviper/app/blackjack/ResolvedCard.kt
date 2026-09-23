package com.cardviper.app.blackjack

import com.cardviper.app.model.CardEventSource
import com.cardviper.app.model.PlayingCard

data class ResolvedCard(
    val rootEventId: String,
    val effectiveEventId: String,
    val card: PlayingCard,
    val source: CardEventSource,
)
