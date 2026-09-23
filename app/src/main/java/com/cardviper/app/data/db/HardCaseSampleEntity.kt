package com.cardviper.app.data.db

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "hard_case_samples",
    indices = [Index(value = ["sessionId"]), Index(value = ["ledgerEventId"]), Index(value = ["reviewId"])],
)
data class HardCaseSampleEntity(
    @PrimaryKey val sampleId: String,
    val sessionId: String,
    val ledgerEventId: String?,
    val reviewId: String?,
    val createdAtEpochMillis: Long,
    val sampleType: String,
    val predictedRank: String?,
    val actualRank: String?,
    val predictedSuit: String?,
    val actualSuit: String?,
    val predictedColor: String?,
    val actualColor: String?,
    val cropReference: String?,
    val exported: Boolean,
)
