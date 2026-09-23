package com.cardviper.app.blackjack

import com.cardviper.app.model.PlayingCard

interface CountStrategy {
    val id: CountStrategyId
    val version: Int

    fun valueOf(card: PlayingCard): Int?

    fun initialRunningCount(deckCount: Int): Int

    fun keyCount(deckCount: Int): Int? = null

    fun insuranceCount(deckCount: Int): Int? = null

    fun evaluate(startingCount: Int, cards: List<PlayingCard>): CountSnapshot {
        var running = startingCount
        var counted = 0
        var unresolved = 0
        cards.forEach { card ->
            val value = valueOf(card)
            if (value == null) {
                unresolved += 1
            } else {
                running += value
                counted += 1
            }
        }
        return CountSnapshot(
            startingCount = startingCount,
            runningCount = running,
            netDelta = running - startingCount,
            cardsCounted = counted,
            unresolvedCards = unresolved,
        )
    }
}
