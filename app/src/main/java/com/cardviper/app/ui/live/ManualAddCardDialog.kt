package com.cardviper.app.ui.live

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
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
import com.cardviper.app.model.CardColor
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardSuit
import com.cardviper.app.model.PlayingCard

@Composable
fun ManualAddCardDialog(
    strategy: CountStrategyId,
    busy: Boolean,
    onDismiss: () -> Unit,
    onAddCard: (PlayingCard) -> Unit,
) {
    var rank by remember { mutableStateOf<CardRank?>(null) }
    var color by remember { mutableStateOf<CardColor?>(null) }
    var suit by remember { mutableStateOf<CardSuit?>(null) }
    var submitted by remember { mutableStateOf(false) }
    val needsTwoColor = strategy == CountStrategyId.KISS_III && rank == CardRank.TWO
    val dialogMaxHeight = (LocalConfiguration.current.screenHeightDp * 0.48f).dp

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Add missed card") },
        text = {
            Column(
                modifier = Modifier.heightIn(max = dialogMaxHeight).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text("Rank")
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    CardRank.entries.forEach { candidate ->
                        FilterChip(
                            selected = rank == candidate,
                            onClick = { rank = candidate },
                            label = { Text(PlayingCard(candidate).shortLabel()) },
                        )
                    }
                }
                Text(if (needsTwoColor) "Color required for KISS III 2" else "Color (optional)")
                if (strategy == CountStrategyId.KO && rank == CardRank.TWO && color == null && suit == null) {
                    Text("Color recommended: needed if this shoe later switches to KISS III")
                }
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    CardColor.entries.forEach { candidate ->
                        FilterChip(
                            selected = color == candidate && suit == null,
                            onClick = {
                                color = candidate
                                suit = null
                            },
                            label = { Text(candidate.name.lowercase().replaceFirstChar { it.uppercase() }) },
                        )
                    }
                }
                Text("Suit (optional)")
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    CardSuit.entries.forEach { candidate ->
                        FilterChip(
                            selected = suit == candidate,
                            onClick = {
                                suit = candidate
                                color = null
                            },
                            label = { Text(PlayingCard(CardRank.ACE, candidate).shortLabel().takeLast(1)) },
                        )
                    }
                }
            }
        },
        confirmButton = {
            TextButton(
                enabled = !busy && !submitted && rank != null && (!needsTwoColor || color != null || suit != null),
                onClick = {
                    val chosenRank = rank ?: return@TextButton
                    submitted = true
                    onAddCard(PlayingCard(chosenRank, suit = suit, colorHint = color))
                    onDismiss()
                },
            ) { Text("ADD CARD") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("CANCEL") } },
    )
}
