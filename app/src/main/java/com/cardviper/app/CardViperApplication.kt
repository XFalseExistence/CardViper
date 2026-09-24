package com.cardviper.app

import android.app.Application
import com.cardviper.app.data.PreferencesRepository
import com.cardviper.app.data.RoomSessionRepository
import com.cardviper.app.data.db.CardViperDatabase
import com.cardviper.app.session.SessionManager
import com.cardviper.app.vision.AndroidImageSourceLoader
import com.cardviper.app.vision.DefaultImageRecognitionEngine
import com.cardviper.app.vision.ImageRecognitionEngine
import com.cardviper.app.vision.ImageSourceLoader
import com.cardviper.app.vision.NoOpCardRecognizer
import com.cardviper.app.vision.UnavailableCardDetector

class CardViperApplication : Application() {
    val database: CardViperDatabase by lazy { CardViperDatabase.get(this) }
    val sessionRepository: RoomSessionRepository by lazy { RoomSessionRepository(database) }
    val sessionManager: SessionManager by lazy { SessionManager(sessionRepository) }
    val preferencesRepository: PreferencesRepository by lazy { PreferencesRepository(this) }
    val imageSourceLoader: ImageSourceLoader by lazy { AndroidImageSourceLoader(this) }
    val imageRecognitionEngine: ImageRecognitionEngine by lazy {
        DefaultImageRecognitionEngine(
            detector = UnavailableCardDetector(),
            recognizer = NoOpCardRecognizer(),
        )
    }
}
