package com.cardviper.app.vision

data class ImageCardObservation(
    val candidate: CardCandidate,
    val cropRect: PixelRect,
    val identity: CardIdentity,
    val classifierConfidence: Float,
    val uncertain: Boolean,
    val alternatives: List<RankedIdentity> = emptyList(),
) {
    init {
        require(classifierConfidence.isFinite() && classifierConfidence in 0f..1f) {
            "Classifier confidence must be finite and between zero and one"
        }
    }
}

data class ImageRecognitionResult(
    val observations: List<ImageCardObservation>,
    val koDelta: Int,
    val kiss3Delta: Int,
    val uncertainCount: Int,
    val skippedCandidates: Int,
) {
    init {
        require(uncertainCount >= 0 && skippedCandidates >= 0) {
            "Result counters cannot be negative"
        }
    }
}
