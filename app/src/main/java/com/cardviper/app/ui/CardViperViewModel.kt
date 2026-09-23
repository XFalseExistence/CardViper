package com.cardviper.app.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.cardviper.app.blackjack.*
import com.cardviper.app.data.*
import com.cardviper.app.model.*
import com.cardviper.app.session.*
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

data class CardViperUiState(
    val snapshot: SessionSnapshot? = null,
    val recentCards: List<ResolvedCard> = emptyList(),
    val preferences: CardViperPreferences = CardViperPreferences(),
    val ready: Boolean = false,
    val busy: Boolean = false,
    val errorMessage: String? = null,
)

/** UI orchestration only. All count changes go through the existing append-only ledger. */
@OptIn(ExperimentalCoroutinesApi::class)
class CardViperViewModel(
    private val manager: SessionManager,
    private val repository: SessionRepository,
    private val preferencesRepository: PreferencesRepository? = null,
) : ViewModel() {
    private val state = MutableStateFlow(CardViperUiState())
    val uiState: StateFlow<CardViperUiState> = state.asStateFlow()
    // A single writer prevents two rapid taps from allocating the same event sequence.
    private val actions = Mutex()

    init {
        viewModelScope.launch {
            try {
                combine(
                    manager.observeSnapshot(),
                    repository.observeActiveSession().flatMapLatest { session ->
                        if (session == null) flowOf(emptyList())
                        else repository.observeEvents(session.sessionId).map { LedgerResolver().resolve(it) }
                    },
                    preferencesRepository?.preferences ?: flowOf(CardViperPreferences()),
                ) { snapshot, cards, preferences -> Triple(snapshot, cards, preferences) }
                    .collect { (snapshot, cards, preferences) ->
                        state.update { it.copy(snapshot = snapshot, recentCards = cards,
                            preferences = preferences, ready = true) }
                    }
            } catch (error: Exception) {
                if (error is CancellationException) throw error
                state.update { it.copy(errorMessage = "Could not load your shoe. Reopen CardViper to retry.") }
            }
        }
    }

    fun startShoe(
        strategyId: CountStrategyId = CountStrategyId.KO,
        decks: Int = 6,
        startMode: StartMode = StartMode.FRESH,
        startingRunningCount: Int? = null,
        startingDeckEstimate: Double? = null,
        onStarted: () -> Unit = {},
    ) = action {
        validateDecks(strategyId, decks)
        require(startMode == StartMode.FRESH || startingRunningCount != null) {
            "Enter a known starting running count for a mid-shoe start."
        }
        require(startingDeckEstimate == null || (startingDeckEstimate.isFinite() &&
            startingDeckEstimate > 0 && startingDeckEstimate <= decks)) {
            "Decks remaining must be greater than zero and no more than the shoe size."
        }
        // Validate before ending the old shoe; its ledger remains untouched.
        if (repository.getActiveSession() != null) manager.end()
        manager.startSession(strategyId, decks, startMode, startingRunningCount,
            if (startMode == StartMode.MID_SHOE) startingDeckEstimate else null)
        refresh()
        onStarted()
    }

    fun manualAdd(card: PlayingCard) = action {
        val session = requireNotNull(repository.getActiveSession()) { "Start a shoe first." }
        require(session.countStrategy != CountStrategyId.KISS_III ||
            card.rank != CardRank.TWO || card.color != null) {
            "Choose red, black, or a suit before adding a 2 in KISS III."
        }
        manager.manualAdd(card)
        refresh()
    }

    fun switchStrategy(strategyId: CountStrategyId) = action {
        val session = requireNotNull(repository.getActiveSession()) { "Start a shoe first." }
        validateDecks(strategyId, session.nominalDecks)
        require(session.startMode != StartMode.MID_SHOE || session.countStrategy == strategyId) {
            "A joined shoe keeps its count mode because earlier cards are unknown. Start a fresh shoe to switch systems."
        }
        // Old KO entries may lack color. Do not silently exclude their twos in KISS III.
        if (strategyId == CountStrategyId.KISS_III) {
            val effective = LedgerResolver().resolve(repository.getEvents(session.sessionId))
            require(effective.none { it.card.rank == CardRank.TWO && it.card.color == null }) {
                "KISS III needs the color of every 2. This shoe has a 2 with unknown color; keep KO for this shoe."
            }
        }
        manager.switchStrategy(strategyId)
        refresh()
    }

    fun undo() = action { manager.undoLast(); refresh() }
    fun pause() = action { manager.pause(); refresh() }
    fun resume(onResumed: () -> Unit = {}) = action { manager.resume(); refresh(); onResumed() }
    fun endShoe(onEnded: () -> Unit) = action { manager.end(); refresh(); onEnded() }

    // Camera lifecycle owns background suspension. A background event is not a shoe reset.
    fun onAppBackgrounded() = Unit
    fun clearError() { state.update { it.copy(errorMessage = null) } }

    fun setShowRecentCards(value: Boolean) = action { preferencesRepository?.setShowRecentCards(value) }

    private fun validateDecks(strategy: CountStrategyId, decks: Int) {
        require(decks in listOf(1, 2, 4, 6, 8)) { "Choose 1, 2, 4, 6, or 8 decks." }
        require(strategy != CountStrategyId.KISS_III || decks == 6) { "KISS III supports six decks in V0.1." }
    }

    private suspend fun refresh() {
        val snapshot = manager.currentSnapshot()
        val cards = snapshot?.let { LedgerResolver().resolve(repository.getEvents(it.session.sessionId)) }
            ?: emptyList()
        state.update { it.copy(snapshot = snapshot, recentCards = cards) }
    }

    private fun action(block: suspend () -> Unit) {
        viewModelScope.launch {
            actions.withLock {
                state.update { it.copy(busy = true, errorMessage = null) }
                try { block() }
                catch (error: Exception) {
                    if (error is CancellationException) throw error
                    state.update { it.copy(errorMessage = if (error is IllegalArgumentException)
                        error.message ?: "Check the selected shoe and card." else "Could not save the change. Please retry.") }
                } finally { state.update { it.copy(busy = false) } }
            }
        }
    }
}

fun modeLabel(mode: CountStrategyId): String = when (mode) {
    CountStrategyId.KO -> "KO"
    CountStrategyId.KISS_III -> "KISS III"
}
