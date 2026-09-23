package com.cardviper.app.data.db

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "card_ledger_events",
    indices = [Index(value = ["sessionId"]), Index(value = ["sessionId", "sequenceNumber"])],
)
data class CardLedgerEventEntity(
    @PrimaryKey val eventId: String,
    val sessionId: String,
    val sequenceNumber: Long,
    val timestampEpochMillis: Long,
    val eventType: String,
    val targetEventId: String?,
    val trackId: Long?,
    val rank: String?,
    val suit: String?,
    val colorHint: String?,
    val source: String,
    val rankConfidence: Float?,
    val colorConfidence: Float?,
    val suitConfidence: Float?,
    val cropReference: String?,
)
