package com.cardviper.app.ui.settings

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.cardviper.app.data.CardViperPreferences

@Composable
fun SettingsScreen(preferences: CardViperPreferences, busy: Boolean,
    onShowRecentCards: (Boolean) -> Unit, onBack: () -> Unit) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)) {
        TextButton(onClick = onBack) { Text("‹ BACK") }
        Text("SETTINGS", style = MaterialTheme.typography.headlineLarge)
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("Show recent cards in HUD", modifier = Modifier.weight(1f))
            Switch(checked = preferences.showRecentCards, enabled = !busy, onCheckedChange = onShowRecentCards)
        }
        Text("Choose KO / KISS III and the deck count in New Shoe.")
        HorizontalDivider()
        Text("VISION", style = MaterialTheme.typography.labelLarge)
        Text("Rear-camera preview only. No automatic card recognition or image collection in this build.")
        Text("CardViper works offline. No network connection is required.",
            style = MaterialTheme.typography.bodySmall)
    }
}
