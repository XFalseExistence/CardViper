package com.cardviper.app.blackjack

class SessionCalculator {
    fun calculate(
        startingCount: Int,
        strategy: CountStrategy,
        resolvedCards: List<ResolvedCard>,
    ): CountSnapshot = strategy.evaluate(startingCount, resolvedCards.map { it.card })
}
