package com.cardviper.app.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface CardViperDao {
    @Query("SELECT * FROM shoe_sessions WHERE state != 'ENDED' ORDER BY createdAtEpochMillis DESC LIMIT 1")
    fun observeActiveSession(): Flow<ShoeSessionEntity?>

    @Query("SELECT * FROM shoe_sessions WHERE state != 'ENDED' ORDER BY createdAtEpochMillis DESC LIMIT 1")
    suspend fun getActiveSession(): ShoeSessionEntity?

    @Query("SELECT * FROM shoe_sessions WHERE sessionId = :sessionId LIMIT 1")
    suspend fun getSession(sessionId: String): ShoeSessionEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertSession(session: ShoeSessionEntity)

    @Query("SELECT * FROM card_ledger_events WHERE sessionId = :sessionId ORDER BY sequenceNumber ASC, eventId ASC")
    fun observeEvents(sessionId: String): Flow<List<CardLedgerEventEntity>>

    @Query("SELECT * FROM card_ledger_events WHERE sessionId = :sessionId ORDER BY sequenceNumber ASC, eventId ASC")
    suspend fun getEvents(sessionId: String): List<CardLedgerEventEntity>

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertEvent(event: CardLedgerEventEntity)

    @Query("SELECT COUNT(*) FROM pending_reviews WHERE sessionId = :sessionId AND state = 'PENDING'")
    fun observePendingReviewCount(sessionId: String): Flow<Int>

    @Query("SELECT COUNT(*) FROM pending_reviews WHERE sessionId = :sessionId AND state = 'PENDING'")
    suspend fun getPendingReviewCount(sessionId: String): Int

    @Query("SELECT * FROM pending_reviews WHERE sessionId = :sessionId ORDER BY createdAtEpochMillis ASC, reviewId ASC")
    fun observePendingReviews(sessionId: String): Flow<List<PendingReviewEntity>>

    @Query("SELECT * FROM pending_reviews WHERE sessionId = :sessionId ORDER BY createdAtEpochMillis ASC, reviewId ASC")
    suspend fun getPendingReviews(sessionId: String): List<PendingReviewEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertPendingReview(review: PendingReviewEntity)

    @Query("UPDATE pending_reviews SET state = :state, updatedAtEpochMillis = :updatedAt WHERE reviewId = :reviewId")
    suspend fun updatePendingReviewState(reviewId: String, state: String, updatedAt: Long)

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertHardCase(sample: HardCaseSampleEntity)
}
