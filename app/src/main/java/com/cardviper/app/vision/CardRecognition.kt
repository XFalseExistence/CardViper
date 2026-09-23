package com.cardviper.app.vision

import com.cardviper.app.model.PlayingCard

data class CardRecognition(
    val card: PlayingCard,
    val rankConfidence: Float,
    val colorConfidence: Float = 0f,
    val suitConfidence: Float = 0f,
)
