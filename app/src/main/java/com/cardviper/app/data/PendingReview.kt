package com.cardviper.app.data

import com.cardviper.app.model.PlayingCard

enum class PendingReviewState {
    PENDING,
    AUTO_RESOLVED,
    MANUALLY_RESOLVED,
    DISCARDED,
}

data class PendingReview(
    val reviewId: String,
    val sessionId: String,
    val trackId: Long? = null,
    val createdAtEpochMillis: Long,
    val updatedAtEpochMillis: Long,
    val bestCard: PlayingCard? = null,
    val rankConfidence: Float? = null,
    val suitConfidence: Float? = null,
    val colorConfidence: Float? = null,
    val cropReference: String? = null,
    val state: PendingReviewState = PendingReviewState.PENDING,
)
