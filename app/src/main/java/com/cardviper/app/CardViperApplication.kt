package com.cardviper.app

import android.app.Application
import com.cardviper.app.data.PreferencesRepository
import com.cardviper.app.data.RoomSessionRepository
import com.cardviper.app.data.db.CardViperDatabase
import com.cardviper.app.session.SessionManager

class CardViperApplication : Application() {
    val database: CardViperDatabase by lazy { CardViperDatabase.get(this) }
    val sessionRepository: RoomSessionRepository by lazy { RoomSessionRepository(database) }
    val sessionManager: SessionManager by lazy { SessionManager(sessionRepository) }
    val preferencesRepository: PreferencesRepository by lazy { PreferencesRepository(this) }
}
