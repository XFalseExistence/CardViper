package com.cardviper.app.session

import com.cardviper.app.blackjack.CountStrategyId

data class ShoeSession(
    val sessionId: String,
    val countStrategy: CountStrategyId,
    val strategyVersion: Int,
    val nominalDecks: Int,
    val startMode: StartMode,
    val startingRunningCount: Int,
    val startingDeckEstimate: Double? = null,
    val state: SessionState,
    val createdAtEpochMillis: Long,
    val startedAtEpochMillis: Long? = null,
    val endedAtEpochMillis: Long? = null,
)
