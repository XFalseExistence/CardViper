package com.cardviper.app.data

import com.cardviper.app.model.CardLedgerEvent
import com.cardviper.app.session.ShoeSession
import kotlinx.coroutines.flow.Flow

interface SessionRepository {
    suspend fun getActiveSession(): ShoeSession?
    suspend fun getSession(sessionId: String): ShoeSession?
    suspend fun getEvents(sessionId: String): List<CardLedgerEvent>
    suspend fun getPendingReviewCount(sessionId: String): Int

    fun observeActiveSession(): Flow<ShoeSession?>
    fun observeEvents(sessionId: String): Flow<List<CardLedgerEvent>>
    fun observePendingReviewCount(sessionId: String): Flow<Int>

    suspend fun upsertSession(session: ShoeSession)
    suspend fun appendEvent(event: CardLedgerEvent)
}
