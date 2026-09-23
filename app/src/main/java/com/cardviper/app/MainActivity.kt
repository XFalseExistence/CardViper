package com.cardviper.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.lifecycle.ViewModelProvider
import com.cardviper.app.ui.CardViperApp
import com.cardviper.app.ui.CardViperViewModel
import com.cardviper.app.ui.theme.CardViperTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val app = application as CardViperApplication
        val viewModel = ViewModelProvider(
            this,
            CardViperViewModel.factory(app.sessionManager),
        )[CardViperViewModel::class.java]

        setContent {
            CardViperTheme {
                CardViperApp(viewModel = viewModel)
            }
        }
    }
}
