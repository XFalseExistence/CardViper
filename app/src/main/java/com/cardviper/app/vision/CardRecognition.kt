package com.cardviper.app.vision

data class CardRecognition(
    val identity: CardIdentity,
    val confidence: Float,
    val alternatives: List<RankedIdentity> = emptyList(),
) {
    init {
        require(confidence in 0f..1f) { "Classifier confidence must be between zero and one" }
    }
}

data class RankedIdentity(val identity: CardIdentity, val confidence: Float) {
    init {
        require(confidence in 0f..1f) { "Alternative confidence must be between zero and one" }
    }
}
