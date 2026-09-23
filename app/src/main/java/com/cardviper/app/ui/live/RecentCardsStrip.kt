package com.cardviper.app.ui.live

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.cardviper.app.blackjack.ResolvedCard
import com.cardviper.app.model.CardRank
import com.cardviper.app.model.CardColor
import com.cardviper.app.model.CardSuit

@Composable
fun RecentCardsStrip(cards: List<ResolvedCard>, onReview: () -> Unit) {
    Row(
        modifier = Modifier.horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        if (cards.isEmpty()) {
            Text("No cards logged", modifier = Modifier.padding(vertical = 8.dp))
        } else {
            cards.takeLast(8).asReversed().forEach { resolved ->
                AssistChip(
                    onClick = onReview,
                    label = { Text(resolved.card.shortLabel()) },
                )
            }
        }
    }
}

internal fun com.cardviper.app.model.PlayingCard.shortLabel(): String {
    val rankLabel = when (rank) {
        CardRank.ACE -> "A"
        CardRank.JACK -> "J"
        CardRank.QUEEN -> "Q"
        CardRank.KING -> "K"
        CardRank.TEN -> "10"
        else -> (rank.ordinal + 1).toString()
    }
    val suitLabel = when (suit) {
        CardSuit.CLUBS -> "♣"
        CardSuit.DIAMONDS -> "♦"
        CardSuit.HEARTS -> "♥"
        CardSuit.SPADES -> "♠"
        null -> when (colorHint) {
            CardColor.RED -> " red"
            CardColor.BLACK -> " black"
            null -> ""
        }
    }
    return rankLabel + suitLabel
}
