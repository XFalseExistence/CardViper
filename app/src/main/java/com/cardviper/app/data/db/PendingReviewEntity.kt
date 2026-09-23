package com.cardviper.app.data.db

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "pending_reviews",
    indices = [Index(value = ["sessionId"]), Index(value = ["trackId"])],
)
data class PendingReviewEntity(
    @PrimaryKey val reviewId: String,
    val sessionId: String,
    val trackId: Long?,
    val createdAtEpochMillis: Long,
    val updatedAtEpochMillis: Long,
    val bestRank: String?,
    val bestSuit: String?,
    val bestColor: String?,
    val rankConfidence: Float?,
    val suitConfidence: Float?,
    val colorConfidence: Float?,
    val cropReference: String?,
    val state: String,
)
