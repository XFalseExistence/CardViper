package com.cardviper.app.ui.review

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.blackjack.ResolvedCard
import com.cardviper.app.data.PendingReview
import com.cardviper.app.model.PlayingCard
import com.cardviper.app.session.SessionSnapshot
import com.cardviper.app.ui.live.shortLabel

@Composable
fun ReviewScreen(
    snapshot: SessionSnapshot?,
    cards: List<ResolvedCard>,
    pendingReviews: List<PendingReview>,
    busy: Boolean,
    onCorrect: (String, PlayingCard) -> Unit,
    onInvalidate: (String) -> Unit,
    onResolvePending: (String, PlayingCard) -> Unit,
    onDiscardPending: (String) -> Unit,
    onBack: () -> Unit,
) {
    var correcting by remember { mutableStateOf<ResolvedCard?>(null) }
    var reviewing by remember { mutableStateOf<PendingReview?>(null) }
    val strategy = snapshot?.session?.countStrategy ?: CountStrategyId.KO

    LazyColumn(
        Modifier.fillMaxSize(),
        contentPadding = PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item { TextButton(onClick = onBack) { Text("‹ BACK") } }
        item { Text("REVIEW", style = MaterialTheme.typography.headlineLarge) }
        item { Text("Active shoe · ${cards.size} cards · ${pendingReviews.size} pending") }
        if (pendingReviews.isNotEmpty()) {
            item { Text("PENDING REVIEW", style = MaterialTheme.typography.titleMedium) }
            items(pendingReviews, key = { it.reviewId }) { pending ->
                Card(
                    modifier = Modifier.fillMaxWidth().clickable(enabled = !busy) { reviewing = pending },
                ) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(pending.bestCard?.shortLabel() ?: "?", style = MaterialTheme.typography.titleLarge)
                        Text("Tap to resolve · does not affect count yet", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
        item { Text("CARD HISTORY", style = MaterialTheme.typography.titleMedium) }
        if (cards.isEmpty()) item { Text("No cards in the active shoe.") }
        items(cards.asReversed(), key = { it.rootEventId }) { resolved ->
            Card(
                modifier = Modifier.fillMaxWidth().clickable(enabled = !busy) { correcting = resolved },
            ) {
                Column(Modifier.padding(16.dp)) {
                    Text(resolved.card.shortLabel(), style = MaterialTheme.typography.titleLarge)
                    Text("${resolved.source.name.lowercase()} · tap to correct", style = MaterialTheme.typography.labelMedium)
                }
            }
        }
    }

    correcting?.let { selected ->
        CardCorrectionSheet(
            strategy = strategy,
            current = selected.card,
            busy = busy,
            onDismiss = { correcting = null },
            onCorrect = { onCorrect(selected.effectiveEventId, it) },
            onRemove = { onInvalidate(selected.effectiveEventId) },
        )
    }
    reviewing?.let { pending ->
        PendingReviewSheet(
            review = pending,
            strategy = strategy,
            busy = busy,
            onDismiss = { reviewing = null },
            onResolve = { onResolvePending(pending.reviewId, it) },
            onDiscard = { onDiscardPending(pending.reviewId) },
        )
    }
}
