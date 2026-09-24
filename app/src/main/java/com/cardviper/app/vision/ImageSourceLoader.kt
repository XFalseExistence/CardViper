package com.cardviper.app.vision

import kotlin.math.roundToInt

fun interface ImageSourceLoader {
    suspend fun load(source: String): VisionImage
}

data class DecodeSize(val width: Int, val height: Int)

/** Preserves aspect ratio to the nearest pixel, without upscaling. */
fun boundedDecodeSize(width: Int, height: Int, maxLongEdge: Int): DecodeSize {
    require(width > 0 && height > 0 && maxLongEdge > 0) { "Image dimensions and decode budget must be positive" }
    val longEdge = maxOf(width, height)
    if (longEdge <= maxLongEdge) return DecodeSize(width, height)
    val scale = maxLongEdge.toDouble() / longEdge
    return DecodeSize((width * scale).roundToInt().coerceAtLeast(1), (height * scale).roundToInt().coerceAtLeast(1))
}

class ImageLoadException(message: String, cause: Throwable? = null) : Exception(message, cause)
class ImageDecodeException(message: String, cause: Throwable? = null) : Exception(message, cause)

/** BitmapFactory accepts power-of-two Int sample sizes; any last pixel is scaled after decode. */
internal fun legacyDecodeSampleSize(width: Int, height: Int, maxLongEdge: Int): Int {
    require(width > 0 && height > 0 && maxLongEdge > 0)
    var sample = 1
    while (sample < (1 shl 30) &&
        ((width - 1) / sample + 1 > maxLongEdge || (height - 1) / sample + 1 > maxLongEdge)) {
        sample *= 2
    }
    return sample
}
