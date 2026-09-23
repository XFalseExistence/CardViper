package com.cardviper.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.cardviper.app.ui.CardViperApp
import com.cardviper.app.ui.CardViperViewModel
import com.cardviper.app.ui.theme.CardViperTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val app = application as CardViperApplication
        val factory = viewModelFactory {
            initializer {
                CardViperViewModel(app.sessionManager, app.sessionRepository, app.preferencesRepository)
            }
        }
        setContent {
            CardViperTheme {
                CardViperApp(viewModel(factory = factory))
            }
        }
    }
}
