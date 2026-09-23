package com.cardviper.app.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.cardviper.app.ui.home.HomeScreen
import com.cardviper.app.ui.live.LiveViperScreen
import com.cardviper.app.ui.navigation.CardViperDestination
import com.cardviper.app.ui.review.ReviewScreen
import com.cardviper.app.ui.settings.SettingsScreen
import com.cardviper.app.ui.shoe.NewShoeScreen

@Composable
fun CardViperApp(viewModel: CardViperViewModel) {
    val navController = rememberNavController()
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    NavHost(
        navController = navController,
        startDestination = CardViperDestination.HOME.route,
    ) {
        composable(CardViperDestination.HOME.route) {
            HomeScreen(
                snapshot = uiState.snapshot,
                onResume = { navController.navigate(CardViperDestination.LIVE.route) },
                onNewShoe = { navController.navigate(CardViperDestination.NEW_SHOE.route) },
                onReview = { navController.navigate(CardViperDestination.REVIEW.route) },
                onSettings = { navController.navigate(CardViperDestination.SETTINGS.route) },
            )
        }

        composable(CardViperDestination.NEW_SHOE.route) {
            NewShoeScreen(
                onBack = { navController.popBackStack() },
                onStart = { strategy, decks, startMode, startingCount, deckEstimate ->
                    viewModel.startShoe(strategy, decks, startMode, startingCount, deckEstimate)
                    navController.navigate(CardViperDestination.LIVE.route) {
                        popUpTo(CardViperDestination.NEW_SHOE.route) { inclusive = true }
                    }
                },
            )
        }

        composable(CardViperDestination.LIVE.route) {
            LiveViperScreen(
                uiState = uiState,
                onSwitchMode = viewModel::switchStrategy,
                onManualAdd = viewModel::manualAdd,
                onUndo = viewModel::undo,
                onReview = { navController.navigate(CardViperDestination.REVIEW.route) },
                onHome = {
                    navController.navigate(CardViperDestination.HOME.route) {
                        popUpTo(CardViperDestination.HOME.route) { inclusive = false }
                        launchSingleTop = true
                    }
                },
            )
        }

        composable(CardViperDestination.REVIEW.route) {
            ReviewScreen(onBack = { navController.popBackStack() })
        }

        composable(CardViperDestination.SETTINGS.route) {
            SettingsScreen(onBack = { navController.popBackStack() })
        }
    }
}
