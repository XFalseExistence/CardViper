package com.cardviper.app.ui.imagetest

/** Source pixels to a centered, aspect-preserving viewport; shared by image and overlays. */
data class FitCenterTransform private constructor(
    val scale: Float,
    val drawnWidth: Float,
    val drawnHeight: Float,
    val offsetX: Float,
    val offsetY: Float,
) {
    fun mapX(sourceX: Float): Float = offsetX + sourceX * scale
    fun mapY(sourceY: Float): Float = offsetY + sourceY * scale

    companion object {
        fun create(sourceWidth: Float, sourceHeight: Float, viewportWidth: Float, viewportHeight: Float): FitCenterTransform {
            require(listOf(sourceWidth, sourceHeight, viewportWidth, viewportHeight).all { it.isFinite() && it > 0f }) {
                "Source and viewport dimensions must be finite and positive"
            }
            val scale = minOf(viewportWidth / sourceWidth, viewportHeight / sourceHeight)
            require(scale.isFinite() && scale > 0f) { "Fit scale must be finite and positive" }
            val width = sourceWidth * scale
            val height = sourceHeight * scale
            return FitCenterTransform(scale, width, height, (viewportWidth - width) / 2f, (viewportHeight - height) / 2f)
        }
    }
}
