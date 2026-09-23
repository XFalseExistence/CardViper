package com.cardviper.app.data.db

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "shoe_sessions")
data class ShoeSessionEntity(
    @PrimaryKey val sessionId: String,
    val countStrategy: String,
    val strategyVersion: Int,
    val nominalDecks: Int,
    val startMode: String,
    val startingRunningCount: Int,
    val startingDeckEstimate: Double?,
    val state: String,
    val createdAtEpochMillis: Long,
    val startedAtEpochMillis: Long?,
    val endedAtEpochMillis: Long?,
)
