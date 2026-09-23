package com.cardviper.app.blackjack

data class CountSnapshot(
    val startingCount: Int,
    val runningCount: Int,
    val netDelta: Int,
    val cardsCounted: Int,
    val unresolvedCards: Int,
)
