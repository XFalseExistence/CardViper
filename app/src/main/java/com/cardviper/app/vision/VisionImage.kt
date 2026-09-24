package com.cardviper.app.vision

/** Source pixel bounds: left/top inclusive, right/bottom exclusive. */
data class PixelRect(val left: Int, val top: Int, val right: Int, val bottom: Int)

/** Packed RGB bytes in row-major order, with three bytes per pixel and no row padding. */
data class VisionImage(val width: Int, val height: Int, val rgb: ByteArray) {
    init {
        require(width > 0 && height > 0) { "Image dimensions must be positive" }
        val pixels = width.toLong() * height.toLong()
        require(pixels <= Int.MAX_VALUE / 3) { "RGB image is too large for a byte array" }
        require(rgb.size == pixels.toInt() * 3) { "RGB buffer length must equal width * height * 3" }
    }

    /** Copies a bounded, nonempty rectangle into an independent packed RGB buffer. */
    fun crop(rect: PixelRect): VisionImage {
        require(rect.left >= 0 && rect.top >= 0 && rect.right <= width && rect.bottom <= height &&
            rect.right > rect.left && rect.bottom > rect.top) { "Crop rectangle must be nonempty and inside the image" }
        val cropWidth = rect.right - rect.left
        val cropHeight = rect.bottom - rect.top
        val rowBytes = cropWidth * 3
        val cropped = ByteArray(rowBytes * cropHeight)
        repeat(cropHeight) { row ->
            val sourceOffset = ((rect.top + row) * width + rect.left) * 3
            rgb.copyInto(cropped, destinationOffset = row * rowBytes,
                startIndex = sourceOffset, endIndex = sourceOffset + rowBytes)
        }
        return VisionImage(cropWidth, cropHeight, cropped)
    }
}
