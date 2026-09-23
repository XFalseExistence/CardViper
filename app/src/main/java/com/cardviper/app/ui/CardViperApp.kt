package com.cardviper.app.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.cardviper.app.ui.home.HomeScreen
import com.cardviper.app.ui.live.LiveViperScreen
import com.cardviper.app.ui.navigation.CardViperDestination.*
import com.cardviper.app.ui.review.ReviewScreen
import com.cardviper.app.ui.settings.SettingsScreen
import com.cardviper.app.ui.shoe.NewShoeScreen

@Composable
fun CardViperApp(model: CardViperViewModel) {
    val state by model.uiState.collectAsStateWithLifecycle()
    val nav = rememberNavController()
    val snackbar = remember { SnackbarHostState() }
    LaunchedEffect(state.errorMessage) {
        state.errorMessage?.let {
            snackbar.showSnackbar(it, withDismissAction = true, duration = SnackbarDuration.Long)
            model.clearError()
        }
    }
    Scaffold(snackbarHost = { SnackbarHost(snackbar) }) { padding ->
        Box(Modifier.fillMaxSize().padding(padding).consumeWindowInsets(padding).imePadding()) {
            if (!state.ready) {
                Column(Modifier.align(Alignment.Center).padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Text("CARDVIPER", style = MaterialTheme.typography.headlineLarge)
                    CircularProgressIndicator()
                    Text("Restoring your shoe…")
                }
            } else NavHost(navController = nav, startDestination = HOME.route) {
                composable(HOME.route) {
                    HomeScreen(state.snapshot, state.busy,
                        onResume = { model.resume { nav.navigate(LIVE.route) { launchSingleTop = true } } },
                        onNewShoe = { nav.navigate(NEW_SHOE.route) { launchSingleTop = true } },
                        onReview = { nav.navigate(REVIEW.route) { launchSingleTop = true } },
                        onSettings = { nav.navigate(SETTINGS.route) { launchSingleTop = true } })
                }
                composable(NEW_SHOE.route) {
                    NewShoeScreen(state.preferences, state.snapshot != null, state.busy,
                        onBack = { nav.popBackStack() },
                        onStart = { strategy, decks, mode, count, estimate ->
                            model.startShoe(strategy, decks, mode, count, estimate) {
                                nav.navigate(LIVE.route) { popUpTo(HOME.route); launchSingleTop = true }
                            }
                        })
                }
                composable(LIVE.route) {
                    val snapshot = state.snapshot
                    if (snapshot != null) LiveViperScreen(
                        snapshot = snapshot,
                        recentCards = state.recentCards,
                        busy = state.busy,
                        showRecentCards = state.preferences.showRecentCards,
                        onBack = { nav.popBackStack(HOME.route, false) },
                        onSwitchStrategy = model::switchStrategy,
                        onAddCard = model::manualAdd,
                        onCorrectCard = model::correctCard,
                        onInvalidateCard = model::invalidateCard,
                        onUndo = model::undo,
                        onPause = model::pause,
                        onResume = { model.resume() },
                        onReview = { nav.navigate(REVIEW.route) { launchSingleTop = true } },
                        onNewShoe = { nav.navigate(NEW_SHOE.route) { launchSingleTop = true } },
                        onEndShoe = { model.endShoe { nav.popBackStack(HOME.route, false) } },
                    ) else LaunchedEffect(Unit) { nav.popBackStack(HOME.route, false) }
                }
                composable(REVIEW.route) {
                    ReviewScreen(
                        snapshot = state.snapshot,
                        cards = state.recentCards,
                        pendingReviews = state.pendingReviews,
                        busy = state.busy,
                        onCorrect = model::correctCard,
                        onInvalidate = model::invalidateCard,
                        onResolvePending = model::resolvePending,
                        onDiscardPending = model::discardPending,
                        onBack = { nav.popBackStack() },
                    )
                }
                composable(SETTINGS.route) {
                    SettingsScreen(state.preferences, state.busy,
                        onShowRecentCards = model::setShowRecentCards,
                        onBack = { nav.popBackStack() })
                }
            }
        }
    }
}
