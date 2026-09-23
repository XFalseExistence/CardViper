package com.cardviper.app.ui.review

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.unit.dp
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.data.PendingReview
import com.cardviper.app.model.CardColor
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.PlayingCard
import com.cardviper.app.ui.live.shortLabel
import java.util.Locale

@Composable
fun PendingReviewSheet(
    review: PendingReview,
    strategy: CountStrategyId,
    busy: Boolean,
    onDismiss: () -> Unit,
    onResolve: (PlayingCard) -> Unit,
    onDiscard: () -> Unit,
) {
    var rank by remember(review.reviewId) { mutableStateOf(review.bestCard?.rank) }
    var suit by remember(review.reviewId) { mutableStateOf(review.bestCard?.suit) }
    var color by remember(review.reviewId) {
        mutableStateOf(if (review.bestCard?.suit == null) review.bestCard?.colorHint else null)
    }
    val needsColor = strategy == CountStrategyId.KISS_III && rank == CardRank.TWO
    val maxHeight = (LocalConfiguration.current.screenHeightDp * 0.52f).dp

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Pending card") },
        text = {
            Column(
                modifier = Modifier.heightIn(max = maxHeight).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    review.bestCard?.let { "Best guess: ${it.shortLabel()}" } ?: "No reliable best guess",
                )
                review.rankConfidence?.let {
                    Text("Rank confidence ${String.format(Locale.US, "%.0f%%", it * 100f)}")
                }
                review.cropReference?.let { Text("Captured review crop available") }
                Text("Rank")
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    CardRank.entries.forEach { option ->
                        FilterChip(
                            selected = rank == option,
                            onClick = { rank = option },
                            label = { Text(PlayingCard(option).shortLabel()) },
                        )
                    }
                }
                Text(if (needsColor) "Color required for KISS III 2" else "Color (optional)")
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    CardColor.entries.forEach { option ->
                        FilterChip(
                            selected = color == option && suit == null,
                            onClick = { color = option; suit = null },
                            label = { Text(option.name.lowercase().replaceFirstChar { it.uppercase() }) },
                        )
                    }
                }
                Text("Suit (optional)")
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    CardSuit.entries.forEach { option ->
                        FilterChip(
                            selected = suit == option,
                            onClick = { suit = option; color = null },
                            label = { Text(PlayingCard(CardRank.ACE, option).shortLabel().takeLast(1)) },
                        )
                    }
                }
                Text("No alternate model candidates are available while recognition is disabled.")
            }
        },
        confirmButton = {
            val selectedRank = rank
            TextButton(
                enabled = !busy && selectedRank != null && (!needsColor || suit != null || color != null),
                onClick = {
                    val chosen = selectedRank ?: return@TextButton
                    onResolve(PlayingCard(chosen, suit = suit, colorHint = color))
                    onDismiss()
                },
            ) { Text("RESOLVE") }
        },
        dismissButton = {
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                TextButton(enabled = !busy, onClick = { onDiscard(); onDismiss() }) { Text("DISCARD") }
                TextButton(onClick = onDismiss) { Text("CANCEL") }
            }
        },
    )
}
