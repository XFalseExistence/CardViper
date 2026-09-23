package com.cardviper.app.tracking

import com.cardviper.app.blackjack.CountStrategy
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.model.CardRank

class CardCommitGate(
    private val minimumStableFrames: Int = 3,
    private val minimumRankConfidence: Float = 0.85f,
    private val minimumColorConfidence: Float = 0.85f,
) {
    fun decide(track: TrackedCard, strategy: CountStrategy): TrackDecision {
        if (track.lifecycle != TrackLifecycle.CONFIRMED) return TrackDecision.NoAction
        if (track.faceState != FaceState.FACE_UP) return TrackDecision.NoAction
        if (track.zone == TableZone.DISCARD) return TrackDecision.NoAction
        if (track.stableFrames < minimumStableFrames) return TrackDecision.Pending("track not stable")

        val card = track.card ?: return TrackDecision.Pending("card identity unresolved")
        if (track.rankConfidence < minimumRankConfidence) return TrackDecision.Pending("rank confidence too low")

        if (strategy.id == CountStrategyId.KISS_III && card.rank == CardRank.TWO) {
            if (card.color == null) return TrackDecision.Pending("KISS III requires 2 color")
            if (track.colorConfidence < minimumColorConfidence) {
                return TrackDecision.Pending("2 color confidence too low")
            }
        }

        if (strategy.valueOf(card) == null) return TrackDecision.Pending("strategy requirements unresolved")

        return TrackDecision.Commit(
            card = card,
            updatedTrack = track.copy(lifecycle = TrackLifecycle.COUNTED),
        )
    }
}
