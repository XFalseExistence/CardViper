package com.cardviper.app.data

import android.content.Context
import androidx.room.Room
import androidx.room.withTransaction
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.cardviper.app.data.db.CardLedgerEventEntity
import com.cardviper.app.data.db.CardViperDatabase
import com.cardviper.app.data.db.ShoeSessionEntity
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.PlayingCard
import com.cardviper.app.session.SessionManager
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CardViperDatabaseTest {
    private lateinit var context: Context
    private val dbName = "cardviper-room-test.db"

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        context.deleteDatabase(dbName)
    }

    @After
    fun tearDown() {
        context.deleteDatabase(dbName)
    }

    @Test
    fun factsSurviveCloseAndReopenAndRecalculateIdentically() = runBlocking {
        var db = openDatabase()
        val repository = RoomSessionRepository(db)
        val manager = SessionManager(repository, idFactory = SequenceIds(), clock = { 10L })
        val session = manager.startSession(com.cardviper.app.blackjack.CountStrategyId.KO, 6)
        manager.manualAdd(PlayingCard(CardRank.FIVE, CardSuit.CLUBS))
        val king = manager.manualAdd(PlayingCard(CardRank.KING, CardSuit.HEARTS))
        manager.correct(king.eventId, PlayingCard(CardRank.QUEEN, CardSuit.HEARTS))
        val seven = manager.manualAdd(PlayingCard(CardRank.SEVEN, CardSuit.SPADES))
        manager.invalidate(seven.eventId)
        val before = manager.currentSnapshot()
        db.close()

        db = openDatabase()
        val recovered = SessionManager(RoomSessionRepository(db), idFactory = SequenceIds(), clock = { 20L })
            .currentSnapshot()

        assertEquals(session.sessionId, recovered?.session?.sessionId)
        assertEquals(before, recovered)
        db.close()
    }

    @Test
    fun failedRoomTransactionDoesNotLeavePartialLedgerFact() = runBlocking {
        val db = openDatabase()
        val dao = db.dao()
        dao.upsertSession(
            ShoeSessionEntity(
                sessionId = "s",
                countStrategy = "KO",
                strategyVersion = 1,
                nominalDecks = 6,
                startMode = "FRESH",
                startingRunningCount = -20,
                startingDeckEstimate = null,
                state = "ACTIVE",
                createdAtEpochMillis = 1,
                startedAtEpochMillis = 1,
                endedAtEpochMillis = null,
            ),
        )

        runCatching {
            db.withTransaction {
                dao.insertEvent(
                    CardLedgerEventEntity(
                        eventId = "event",
                        sessionId = "s",
                        sequenceNumber = 1,
                        timestampEpochMillis = 1,
                        eventType = "CARD_COMMITTED",
                        targetEventId = null,
                        trackId = null,
                        rank = "FIVE",
                        suit = "CLUBS",
                        colorHint = null,
                        source = "VISION",
                        rankConfidence = 1f,
                        colorConfidence = 1f,
                        suitConfidence = 1f,
                        cropReference = null,
                    ),
                )
                error("force rollback")
            }
        }

        assertEquals(emptyList<CardLedgerEventEntity>(), dao.getEvents("s"))
        db.close()
    }

    private fun openDatabase() = Room.databaseBuilder(context, CardViperDatabase::class.java, dbName)
        .allowMainThreadQueries()
        .build()

    private class SequenceIds : com.cardviper.app.session.IdFactory {
        private var next = 0
        override fun newId(): String = "id-${next++}"
    }
}
