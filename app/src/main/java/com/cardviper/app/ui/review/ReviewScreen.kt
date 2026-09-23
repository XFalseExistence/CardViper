package com.cardviper.app.ui.review

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.cardviper.app.blackjack.ResolvedCard
import com.cardviper.app.session.SessionSnapshot
import com.cardviper.app.ui.live.shortLabel

/** Read-only shell. Editing and pending-candidate resolution belong to Task 7. */
@Composable
fun ReviewScreen(snapshot: SessionSnapshot?, cards: List<ResolvedCard>, onBack: () -> Unit) {
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { TextButton(onClick = onBack) { Text("‹ BACK") } }
        item { Text("REVIEW", style = MaterialTheme.typography.headlineLarge) }
        item { Text("Active shoe · ${cards.size} cards · ${snapshot?.pendingReviews ?: 0} pending") }
        item { Text("Card history is read-only in this build. Corrections, pending review, and archived shoe browsing come next.",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
        if (cards.isEmpty()) item { Text("No cards in the active shoe.") }
        items(cards.asReversed(), key = { it.rootEventId }) { resolved ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text(resolved.card.shortLabel(), style = MaterialTheme.typography.titleLarge)
                    Text(resolved.source.name.lowercase(), style = MaterialTheme.typography.labelMedium)
                }
            }
        }
    }
}
