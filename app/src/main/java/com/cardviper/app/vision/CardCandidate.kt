package com.cardviper.app.vision

/**
 * Detector geometry in source-image pixels, never normalized coordinates.
 * Finite boxes may cross image edges or have nonpositive area. The crop stage
 * clamps out-of-bounds boxes and skips degenerate candidates.
 */
data class CardCandidate(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val confidence: Float,
) {
    init {
        require(x.isFinite() && y.isFinite() && width.isFinite() && height.isFinite()) {
            "Candidate geometry must contain finite source-image pixel coordinates"
        }
        require(confidence in 0f..1f) { "Detector confidence must be between zero and one" }
    }
}
