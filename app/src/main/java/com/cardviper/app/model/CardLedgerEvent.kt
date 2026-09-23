package com.cardviper.app.model

data class CardLedgerEvent(
    val eventId: String,
    val sessionId: String,
    val sequenceNumber: Long,
    val timestampEpochMillis: Long,
    val eventType: LedgerEventType,
    val targetEventId: String? = null,
    val trackId: Long? = null,
    val card: PlayingCard? = null,
    val source: CardEventSource,
    val rankConfidence: Float? = null,
    val colorConfidence: Float? = null,
    val suitConfidence: Float? = null,
    val cropReference: String? = null,
)
