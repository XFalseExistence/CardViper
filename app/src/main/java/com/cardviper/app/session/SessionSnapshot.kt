package com.cardviper.app.session

data class SessionSnapshot(
    val session: ShoeSession,
    val runningCount: Int,
    val cardsSeen: Int,
    val penetration: Double,
    val estimatedDecksRemaining: Double,
    val keyCount: Int?,
    val insuranceCount: Int?,
    val pendingReviews: Int,
)
