package com.cardviper.app.tracking

import com.cardviper.app.blackjack.Kiss3Strategy
import com.cardviper.app.blackjack.KoStrategy
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.PlayingCard
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CardCommitGateTest {
    private val gate = CardCommitGate()

    @Test
    fun samePhysicalCardObservedOneHundredTimesCommitsExactlyOnce() {
        var track = confirmedTrack(42, PlayingCard(CardRank.KING, CardSuit.HEARTS))
        var commits = 0

        repeat(100) {
            when (val decision = gate.decide(track, KoStrategy())) {
                is TrackDecision.Commit -> {
                    commits += 1
                    track = decision.updatedTrack
                }
                else -> Unit
            }
        }

        assertEquals(1, commits)
        assertEquals(TrackLifecycle.COUNTED, track.lifecycle)
    }

    @Test
    fun reacquiredCountedTrackDoesNotCommitAgainButNewIdenticalTrackCan() {
        val card = PlayingCard(CardRank.KING, CardSuit.HEARTS)
        val first = gate.decide(confirmedTrack(10, card), KoStrategy()) as TrackDecision.Commit
        val reacquired = first.updatedTrack.copy(missingFrames = 4, lastSeenEpochMillis = 500)

        assertEquals(TrackDecision.NoAction, gate.decide(reacquired, KoStrategy()))
        assertTrue(gate.decide(confirmedTrack(11, card), KoStrategy()) is TrackDecision.Commit)
    }

    @Test
    fun faceDownCardWaitsUntilSameTrackIsRevealed() {
        val hidden = confirmedTrack(7, PlayingCard(CardRank.QUEEN, CardSuit.SPADES)).copy(faceState = FaceState.FACE_DOWN)
        assertEquals(TrackDecision.NoAction, gate.decide(hidden, KoStrategy()))

        val revealed = hidden.copy(faceState = FaceState.FACE_UP)
        val commit = gate.decide(revealed, KoStrategy()) as TrackDecision.Commit
        assertEquals(7, commit.updatedTrack.trackId)
        assertEquals(TrackLifecycle.COUNTED, commit.updatedTrack.lifecycle)
    }

    @Test
    fun kissThreeTwoWithUnknownColorIsPendingUntilColorResolves() {
        val unresolved = confirmedTrack(9, PlayingCard(CardRank.TWO))
        val pending = gate.decide(unresolved, Kiss3Strategy())
        assertTrue(pending is TrackDecision.Pending)

        val resolved = unresolved.copy(card = PlayingCard(CardRank.TWO, CardSuit.CLUBS), colorConfidence = 0.98f)
        val commit = gate.decide(resolved, Kiss3Strategy()) as TrackDecision.Commit
        assertEquals(TrackLifecycle.COUNTED, commit.updatedTrack.lifecycle)
    }

    @Test
    fun lowRankConfidenceRemainsPendingAndNeverCommits() {
        val weak = confirmedTrack(12, PlayingCard(CardRank.SEVEN, CardSuit.CLUBS)).copy(rankConfidence = 0.55f)
        assertTrue(gate.decide(weak, KoStrategy()) is TrackDecision.Pending)
    }

    private fun confirmedTrack(trackId: Long, card: PlayingCard) = TrackedCard(
        trackId = trackId,
        lifecycle = TrackLifecycle.CONFIRMED,
        faceState = FaceState.FACE_UP,
        zone = TableZone.PLAYER,
        firstSeenEpochMillis = 100,
        lastSeenEpochMillis = 200,
        stableFrames = 4,
        missingFrames = 0,
        card = card,
        rankConfidence = 0.96f,
        colorConfidence = 0.96f,
        suitConfidence = 0.90f,
    )
}
