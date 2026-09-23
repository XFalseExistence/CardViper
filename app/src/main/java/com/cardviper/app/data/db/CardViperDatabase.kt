package com.cardviper.app.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [
        ShoeSessionEntity::class,
        CardLedgerEventEntity::class,
        PendingReviewEntity::class,
        HardCaseSampleEntity::class,
    ],
    version = 1,
    exportSchema = false,
)
abstract class CardViperDatabase : RoomDatabase() {
    abstract fun dao(): CardViperDao

    companion object {
        @Volatile private var instance: CardViperDatabase? = null

        fun get(context: Context): CardViperDatabase = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(
                context.applicationContext,
                CardViperDatabase::class.java,
                "cardviper.db",
            ).build().also { instance = it }
        }
    }
}
