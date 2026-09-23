package com.cardviper.app.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.cardviper.app.blackjack.CountStrategyId
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.cardViperDataStore by preferencesDataStore(name = "cardviper_preferences")

class PreferencesRepository(
    private val context: Context,
) {
    val preferences: Flow<CardViperPreferences> = context.cardViperDataStore.data.map { values ->
        CardViperPreferences(
            defaultCountMode = values[DEFAULT_COUNT]
                ?.let { runCatching { CountStrategyId.valueOf(it) }.getOrNull() }
                ?: CountStrategyId.KO,
            defaultDecks = values[DEFAULT_DECKS] ?: 6,
            autoAttention = values[AUTO_ATTENTION] ?: true,
            saveCorrectionCrops = values[SAVE_CORRECTION_CROPS] ?: true,
            saveUncertainCrops = values[SAVE_UNCERTAIN_CROPS] ?: true,
            showConfidence = values[SHOW_CONFIDENCE] ?: false,
            showRecentCards = values[SHOW_RECENT_CARDS] ?: true,
            visionDebug = values[VISION_DEBUG] ?: false,
        )
    }

    suspend fun setDefaultCountMode(value: CountStrategyId) {
        context.cardViperDataStore.edit { it[DEFAULT_COUNT] = value.name }
    }

    suspend fun setDefaultDecks(value: Int) {
        require(value > 0) { "defaultDecks must be positive" }
        context.cardViperDataStore.edit { it[DEFAULT_DECKS] = value }
    }

    suspend fun setAutoAttention(value: Boolean) {
        context.cardViperDataStore.edit { it[AUTO_ATTENTION] = value }
    }

    suspend fun setSaveCorrectionCrops(value: Boolean) {
        context.cardViperDataStore.edit { it[SAVE_CORRECTION_CROPS] = value }
    }

    suspend fun setSaveUncertainCrops(value: Boolean) {
        context.cardViperDataStore.edit { it[SAVE_UNCERTAIN_CROPS] = value }
    }

    suspend fun setShowConfidence(value: Boolean) {
        context.cardViperDataStore.edit { it[SHOW_CONFIDENCE] = value }
    }

    suspend fun setShowRecentCards(value: Boolean) {
        context.cardViperDataStore.edit { it[SHOW_RECENT_CARDS] = value }
    }

    suspend fun setVisionDebug(value: Boolean) {
        context.cardViperDataStore.edit { it[VISION_DEBUG] = value }
    }

    private companion object {
        val DEFAULT_COUNT = stringPreferencesKey("default_count")
        val DEFAULT_DECKS = intPreferencesKey("default_decks")
        val AUTO_ATTENTION = booleanPreferencesKey("auto_attention")
        val SAVE_CORRECTION_CROPS = booleanPreferencesKey("save_correction_crops")
        val SAVE_UNCERTAIN_CROPS = booleanPreferencesKey("save_uncertain_crops")
        val SHOW_CONFIDENCE = booleanPreferencesKey("show_confidence")
        val SHOW_RECENT_CARDS = booleanPreferencesKey("show_recent_cards")
        val VISION_DEBUG = booleanPreferencesKey("vision_debug")
    }
}
