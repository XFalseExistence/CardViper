package com.cardviper.app.data

import com.cardviper.app.model.PlayingCard

enum class HardCaseType {
    WRONG_RANK,
    WRONG_SUIT,
    WRONG_COLOR,
    FALSE_POSITIVE,
    LOW_CONFIDENCE,
    MANUAL_MISSED_CARD,
    TRACKING_FAILURE,
}

data class HardCaseSample(
    val sampleId: String,
    val sessionId: String,
    val ledgerEventId: String? = null,
    val reviewId: String? = null,
    val createdAtEpochMillis: Long,
    val sampleType: HardCaseType,
    val predictedCard: PlayingCard? = null,
    val actualCard: PlayingCard? = null,
    val cropReference: String? = null,
    val exported: Boolean = false,
)
