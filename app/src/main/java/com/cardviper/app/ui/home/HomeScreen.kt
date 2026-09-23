package com.cardviper.app.ui.home

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.cardviper.app.session.SessionSnapshot
import com.cardviper.app.ui.modeLabel

@Composable
fun HomeScreen(snapshot: SessionSnapshot?, busy: Boolean, onResume: () -> Unit,
    onNewShoe: () -> Unit, onReview: () -> Unit, onSettings: () -> Unit) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally) {
        Spacer(Modifier.height(28.dp))
        Text("♠", fontSize = 64.sp, color = MaterialTheme.colorScheme.primary)
        Text("CARDVIPER", style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Black, letterSpacing = 3.sp)
        Text("KNOW • COUNT • PLAY SMART", style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.secondary)
        Spacer(Modifier.height(20.dp))
        if (snapshot != null) Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("YOUR SHOE", style = MaterialTheme.typography.labelMedium)
                Text("${modeLabel(snapshot.session.countStrategy)} · ${snapshot.session.nominalDecks} decks")
                Text("Running count ${snapshot.runningCount} · ${snapshot.cardsSeen} cards seen")
                Button(onClick = onResume, enabled = !busy, modifier = Modifier.fillMaxWidth()) { Text("RESUME SHOE") }
            }
        }
        Button(onClick = onNewShoe, enabled = !busy,
            modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp)) { Text("START NEW SHOE") }
        OutlinedButton(onClick = onReview, modifier = Modifier.fillMaxWidth()) { Text("REVIEW") }
        OutlinedButton(onClick = onSettings, modifier = Modifier.fillMaxWidth()) { Text("SETTINGS") }
        Spacer(Modifier.height(12.dp))
        Text("V0.1 • Offline training", style = MaterialTheme.typography.labelLarge)
        Text("Live camera preview and manual counting. Automatic card recognition is not active.",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
