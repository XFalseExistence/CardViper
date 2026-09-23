package com.cardviper.app.vision

data class CardCandidate(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val confidence: Float,
)
