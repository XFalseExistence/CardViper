package com.cardviper.app.session

import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.data.CardViperPreferences
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CardViperPreferencesTest {
    @Test
    fun defaultsMatchLockedV0OneDesign() {
        val prefs = CardViperPreferences()
        assertEquals(CountStrategyId.KO, prefs.defaultCountMode)
        assertEquals(6, prefs.defaultDecks)
        assertTrue(prefs.autoAttention)
        assertTrue(prefs.saveCorrectionCrops)
        assertTrue(prefs.saveUncertainCrops)
        assertFalse(prefs.showConfidence)
        assertTrue(prefs.showRecentCards)
        assertFalse(prefs.visionDebug)
    }
}
