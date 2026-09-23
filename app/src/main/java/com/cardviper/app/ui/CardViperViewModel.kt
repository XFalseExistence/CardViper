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
    val pendingReviews: List<PendingReview> = emptyList(),
    val preferences: CardViperPreferences = CardViperPreferences(),
    val ready: Boolean = false,
    val busy: Boolean = false,
    val errorMessage: String? = null,
)

@OptIn(ExperimentalCoroutinesApi::class)
class CardViperViewModel(
    private val manager: SessionManager,
    private val repository: SessionRepository,
    private val preferencesRepository: PreferencesRepository? = null,
) : ViewModel() {
    private val state = MutableStateFlow(CardViperUiState())
    val uiState: StateFlow<CardViperUiState> = state.asStateFlow()
    private val actions = Mutex()

    init {
        viewModelScope.launch {
            try {
                val active = repository.observeActiveSession()
                combine(
                    manager.observeSnapshot(),
                    active.flatMapLatest { session ->
                        if (session == null) flowOf(emptyList())
                        else repository.observeEvents(session.sessionId).map { LedgerResolver().resolve(it) }
                    },
                    active.flatMapLatest { session ->
                        if (session == null) flowOf(emptyList()) else repository.observePendingReviews(session.sessionId)
                    },
                    preferencesRepository?.preferences ?: flowOf(CardViperPreferences()),
                ) { snapshot, cards, pending, preferences ->
                    CardViperUiState(snapshot, cards, pending, preferences, ready = true)
                }.collect { incoming ->
                    state.update { current -> incoming.copy(busy = current.busy, errorMessage = current.errorMessage) }
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
        require(startMode == StartMode.FRESH || startingRunningCount != null) { "Enter a known starting running count for a mid-shoe start." }
        require(startingDeckEstimate == null || (startingDeckEstimate.isFinite() && startingDeckEstimate > 0 && startingDeckEstimate <= decks)) {
            "Decks remaining must be greater than zero and no more than the shoe size."
        }
        if (repository.getActiveSession() != null) manager.end()
        manager.startSession(strategyId, decks, startMode, startingRunningCount, if (startMode == StartMode.MID_SHOE) startingDeckEstimate else null)
        refresh()
        onStarted()
    }

    fun manualAdd(card: PlayingCard) = action {
        val session = requireNotNull(repository.getActiveSession()) { "Start a shoe first." }
        require(session.countStrategy != CountStrategyId.KISS_III || card.rank != CardRank.TWO || card.color != null) {
            "Choose red, black, or a suit before adding a 2 in KISS III."
        }
        manager.manualAdd(card)
        refresh()
    }

    fun correctCard(targetEventId: String, card: PlayingCard) = action {
        manager.correctWithHardCase(targetEventId, card)
        refresh()
    }

    fun invalidateCard(targetEventId: String) = action {
        manager.invalidateAsFalsePositive(targetEventId)
        refresh()
    }

    fun resolvePending(reviewId: String, card: PlayingCard) = action {
        manager.resolvePendingReview(reviewId, card)
        refresh()
    }

    fun discardPending(reviewId: String) = action {
        manager.discardPendingReview(reviewId)
        refresh()
    }

    fun switchStrategy(strategyId: CountStrategyId) = action {
        val session = requireNotNull(repository.getActiveSession()) { "Start a shoe first." }
        validateDecks(strategyId, session.nominalDecks)
        require(session.startMode != StartMode.MID_SHOE || session.countStrategy == strategyId) {
            "A joined shoe keeps its count mode because earlier cards are unknown. Start a fresh shoe to switch systems."
        }
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
    fun onAppBackgrounded() = Unit
    fun clearError() { state.update { it.copy(errorMessage = null) } }
    fun setShowRecentCards(value: Boolean) = action { preferencesRepository?.setShowRecentCards(value) }

    private fun validateDecks(strategy: CountStrategyId, decks: Int) {
        require(decks in listOf(1, 2, 4, 6, 8)) { "Choose 1, 2, 4, 6, or 8 decks." }
        require(strategy != CountStrategyId.KISS_III || decks == 6) { "KISS III supports six decks in V0.1." }
    }

    private suspend fun refresh() {
        val snapshot = manager.currentSnapshot()
        val cards = snapshot?.let { LedgerResolver().resolve(repository.getEvents(it.session.sessionId)) } ?: emptyList()
        val pending = snapshot?.let { repository.getPendingReviews(it.session.sessionId) } ?: emptyList()
        state.update { it.copy(snapshot = snapshot, recentCards = cards, pendingReviews = pending) }
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
