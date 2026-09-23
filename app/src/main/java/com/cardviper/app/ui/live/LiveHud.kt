package com.cardviper.app.ui.live

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.blackjack.ResolvedCard
import com.cardviper.app.session.SessionSnapshot
import com.cardviper.app.session.SessionState
import com.cardviper.app.session.StartMode
import java.util.Locale

private val Gold = Color(0xFFE5BF70)

@Composable
fun LiveHud(
    snapshot: SessionSnapshot,
    recentCards: List<ResolvedCard>,
    busy: Boolean,
    showRecentCards: Boolean,
    onSwitchStrategy: (CountStrategyId) -> Unit,
    onAddCard: () -> Unit,
    onUndo: () -> Unit,
    onReview: () -> Unit,
    onCorrectCard: (ResolvedCard) -> Unit,
) {
    val session = snapshot.session
    val editable = !busy && session.state != SessionState.ENDED
    Column(
        modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(9.dp),
    ) {
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            FilterChip(
                selected = session.countStrategy == CountStrategyId.KO,
                onClick = { onSwitchStrategy(CountStrategyId.KO) },
                enabled = editable && session.startMode != StartMode.MID_SHOE && session.countStrategy != CountStrategyId.KO,
                label = { Text("KO") },
            )
            FilterChip(
                selected = session.countStrategy == CountStrategyId.KISS_III,
                onClick = { onSwitchStrategy(CountStrategyId.KISS_III) },
                enabled = editable && session.startMode != StartMode.MID_SHOE &&
                    session.nominalDecks == 6 && session.countStrategy != CountStrategyId.KISS_III,
                label = { Text("KISS III") },
            )
        }
        if (session.nominalDecks != 6) Text("KISS III uses the six-deck profile", color = Gold, fontSize = 12.sp)
        if (session.startMode == StartMode.MID_SHOE) Text("Joined shoe: count mode fixed because earlier cards are unknown.", color = Gold, fontSize = 12.sp)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(18.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Column {
                Text("RUNNING COUNT", fontSize = 11.sp, letterSpacing = 1.4.sp, color = Gold)
                Text(snapshot.runningCount.toString(), fontSize = 56.sp, lineHeight = 58.sp, fontWeight = FontWeight.Bold, color = Color(0xFF31F58A))
            }
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("${if (session.startMode == StartMode.FRESH) "IRC" else "START RC"}  ${session.startingRunningCount}", color = Gold)
                snapshot.keyCount?.let { Text("KEY  $it", color = Gold) }
                snapshot.insuranceCount?.let { Text("INSURANCE  $it", color = Gold) }
            }
        }
        Text(
            "${session.nominalDecks} decks  ·  ${String.format(Locale.US, "%.1f", snapshot.estimatedDecksRemaining)} left" +
                "  ·  ${String.format(Locale.US, "%.0f", snapshot.penetration * 100)}% seen  ·  ${snapshot.cardsSeen} cards",
            fontSize = 13.sp,
        )
        if (showRecentCards) {
            Text("RECENT CARDS · tap to correct", fontSize = 11.sp, letterSpacing = 1.4.sp, color = Gold)
            RecentCardsStrip(recentCards, onCorrectCard)
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Button(onClick = onAddCard, enabled = editable) { Text("+ CARD") }
            OutlinedButton(onClick = onUndo, enabled = editable && snapshot.cardsSeen > 0) { Text("UNDO") }
            OutlinedButton(onClick = onReview) { Text("REVIEW ${snapshot.pendingReviews}") }
        }
    }
}
