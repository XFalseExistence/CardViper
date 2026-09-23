package com.cardviper.app.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.cardviper.app.blackjack.CountStrategyId
import com.cardviper.app.blackjack.ResolvedCard
import com.cardviper.app.model.PlayingCard
import com.cardviper.app.session.SessionManager
import com.cardviper.app.session.SessionSnapshot
import com.cardviper.app.session.StartMode
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class CardViperUiState(
    val snapshot: SessionSnapshot? = null,
    val recentCards: List<ResolvedCard> = emptyList(),
    val errorMessage: String? = null,
)

class CardViperViewModel(
    private val sessionManager: SessionManager,
) : ViewModel() {
    private val error = MutableStateFlow<String?>(null)

    val uiState: StateFlow<CardViperUiState> = combine(
        sessionManager.observeSnapshot(),
        sessionManager.observeEffectiveCards(),
        error,
    ) { snapshot, cards, message ->
        CardViperUiState(
            snapshot = snapshot,
            recentCards = cards.takeLast(12),
            errorMessage = message,
        )
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.Eagerly,
        initialValue = CardViperUiState(),
    )

    fun startShoe(
        strategyId: CountStrategyId = CountStrategyId.KO,
        decks: Int = 6,
        startMode: StartMode = StartMode.FRESH,
        startingRunningCount: Int? = null,
        startingDeckEstimate: Double? = null,
    ) = launchAction {
        sessionManager.startSession(
            strategyId = strategyId,
            nominalDecks = decks,
            startMode = startMode,
            startingRunningCount = startingRunningCount,
            startingDeckEstimate = startingDeckEstimate,
        )
    }

    fun switchStrategy(strategyId: CountStrategyId) = launchAction {
        sessionManager.switchStrategy(strategyId)
    }

    fun manualAdd(card: PlayingCard) = launchAction {
        sessionManager.manualAdd(card)
    }

    fun undo() = launchAction {
        sessionManager.undoLast()
    }

    fun pause() = launchAction { sessionManager.pause() }

    fun resume() = launchAction { sessionManager.resume() }

    fun endShoe() = launchAction { sessionManager.end() }

    fun clearError() {
        error.value = null
    }

    fun onAppBackgrounded() {
        // Deliberately no session mutation. Android may background the UI without ending a shoe.
    }

    private fun launchAction(block: suspend () -> Unit) {
        viewModelScope.launch {
            runCatching { block() }
                .onSuccess { error.value = null }
                .onFailure { error.value = it.message ?: "CardViper action failed" }
        }
    }

    companion object {
        fun factory(sessionManager: SessionManager): ViewModelProvider.Factory =
            object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T {
                    require(modelClass.isAssignableFrom(CardViperViewModel::class.java))
                    return CardViperViewModel(sessionManager) as T
                }
            }
    }
}
