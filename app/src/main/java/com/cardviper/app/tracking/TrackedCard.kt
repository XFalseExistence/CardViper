package com.cardviper.app.tracking

import com.cardviper.app.model.PlayingCard

data class TrackedCard(
    val trackId: Long,
    val lifecycle: TrackLifecycle,
    val faceState: FaceState,
    val zone: TableZone,
    val firstSeenEpochMillis: Long,
    val lastSeenEpochMillis: Long,
    val stableFrames: Int,
    val missingFrames: Int,
    val card: PlayingCard? = null,
    val rankConfidence: Float = 0f,
    val colorConfidence: Float = 0f,
    val suitConfidence: Float = 0f,
)
