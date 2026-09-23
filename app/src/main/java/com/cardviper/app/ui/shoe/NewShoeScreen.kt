package com.cardviper.app.ui.shoe

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.data.CardViperPreferences
import com.cardviper.app.session.StartMode
import com.cardviper.app.ui.modeLabel

@Composable
fun NewShoeScreen(preferences: CardViperPreferences, hasActiveShoe: Boolean, busy: Boolean,
    onBack: () -> Unit, onStart: (CountStrategyId, Int, StartMode, Int?, Double?) -> Unit) {
    var strategy by rememberSaveable { mutableStateOf(preferences.defaultCountMode) }
    var decks by rememberSaveable { mutableIntStateOf(if (strategy == CountStrategyId.KISS_III) 6
        else preferences.defaultDecks.takeIf { it in listOf(1, 2, 4, 6, 8) } ?: 6) }
    var startMode by rememberSaveable { mutableStateOf(StartMode.FRESH) }
    var count by rememberSaveable { mutableStateOf("") }
    var remaining by rememberSaveable { mutableStateOf("") }
    var confirmReplacement by rememberSaveable { mutableStateOf(false) }
    val validCount = startMode == StartMode.FRESH || count.toIntOrNull() != null
    val remainingValue = remaining.toDoubleOrNull()
    val validRemaining = startMode == StartMode.FRESH || remaining.isBlank() ||
        (remainingValue != null && remainingValue.isFinite() && remainingValue > 0 && remainingValue <= decks)
    val start = { onStart(strategy, decks, startMode,
        if (startMode == StartMode.MID_SHOE) count.toIntOrNull() else null,
        if (startMode == StartMode.MID_SHOE) remainingValue else null) }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)) {
        TextButton(onClick = onBack) { Text("‹ BACK") }
        Text("NEW SHOE", style = MaterialTheme.typography.headlineLarge)
        Text("COUNTING SYSTEM", style = MaterialTheme.typography.labelLarge)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            CountStrategyId.entries.forEach { mode ->
                FilterChip(selected = strategy == mode, enabled = !busy,
                    onClick = { strategy = mode; if (mode == CountStrategyId.KISS_III) decks = 6 },
                    label = { Text(modeLabel(mode)) })
            }
        }
        Text("TOTAL DECKS", style = MaterialTheme.typography.labelLarge)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(1, 2, 4, 6, 8).forEach { value ->
                FilterChip(selected = decks == value,
                    enabled = !busy && (strategy == CountStrategyId.KO || value == 6),
                    onClick = { decks = value }, label = { Text(value.toString()) })
            }
        }
        if (strategy == CountStrategyId.KISS_III) Text("KISS III supports six decks in V0.1.",
            style = MaterialTheme.typography.bodySmall)
        Text("START MODE", style = MaterialTheme.typography.labelLarge)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(selected = startMode == StartMode.FRESH, enabled = !busy,
                onClick = { startMode = StartMode.FRESH }, label = { Text("Fresh shoe") })
            FilterChip(selected = startMode == StartMode.MID_SHOE, enabled = !busy,
                onClick = { startMode = StartMode.MID_SHOE }, label = { Text("Join mid-shoe") })
        }
        if (startMode == StartMode.MID_SHOE) {
            Text("Enter a known running count for the selected system.")
            OutlinedTextField(value = count, onValueChange = { count = it },
                label = { Text("Starting running count (required)") }, singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text),
                isError = count.isNotBlank() && !validCount, modifier = Modifier.fillMaxWidth())
            OutlinedTextField(value = remaining, onValueChange = { remaining = it },
                label = { Text("Decks remaining (optional)") }, singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                isError = !validRemaining, modifier = Modifier.fillMaxWidth())
            Text("Without an estimate, deck context starts at the total deck count.",
                style = MaterialTheme.typography.bodySmall)
        }
        Text("Camera access is requested when you start. You can count manually without it.",
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        Button(onClick = { if (hasActiveShoe) confirmReplacement = true else start() },
            enabled = !busy && validCount && validRemaining,
            modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp)) { Text("START VIPER") }
    }
    if (confirmReplacement) AlertDialog(onDismissRequest = { confirmReplacement = false },
        title = { Text("Start a new shoe?") },
        text = { Text("The current shoe will end. Its ledger history remains saved.") },
        confirmButton = { TextButton(enabled = !busy, onClick = {
            confirmReplacement = false; start()
        }) { Text("START NEW SHOE") } },
        dismissButton = { TextButton(onClick = { confirmReplacement = false }) { Text("CANCEL") } })
}
