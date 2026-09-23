package com.cardviper.app.data

import com.cardviper.app.blackjack.CountStrategyId

data class CardViperPreferences(
    val defaultCountMode: CountStrategyId = CountStrategyId.KO,
    val defaultDecks: Int = 6,
    val autoAttention: Boolean = true,
    val saveCorrectionCrops: Boolean = true,
    val saveUncertainCrops: Boolean = true,
    val showConfidence: Boolean = false,
    val showRecentCards: Boolean = true,
    val visionDebug: Boolean = false,
)
