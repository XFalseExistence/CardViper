package com.cardviper.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val CardViperGreen = Color(0xFF31F58A)
private val CardViperBackground = Color(0xFF050807)
private val CardViperSurface = Color(0xFF101512)

private val CardViperDarkColors = darkColorScheme(
    primary = CardViperGreen,
    secondary = Color(0xFF8DEDB7),
    background = CardViperBackground,
    surface = CardViperSurface,
    onPrimary = Color.Black,
    onBackground = Color(0xFFE8F2EC),
    onSurface = Color(0xFFE8F2EC),
)

@Composable
fun CardViperTheme(
    @Suppress("UNUSED_PARAMETER") darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = CardViperDarkColors,
        content = content,
    )
}
