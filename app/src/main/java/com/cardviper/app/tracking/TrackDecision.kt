package com.cardviper.app.tracking

import com.cardviper.app.model.PlayingCard

sealed interface TrackDecision {
    data class Commit(
        val card: PlayingCard,
        val updatedTrack: TrackedCard,
    ) : TrackDecision

    data class Pending(val reason: String) : TrackDecision

    data object NoAction : TrackDecision
}
