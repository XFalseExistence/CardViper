package com.cardviper.app.ui.imagetest

import android.graphics.Bitmap
import com.cardviper.app.vision.VisionImage
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext

/** Display the exact packed RGB geometry used by inference; never decode the URI again. */
internal suspend fun VisionImage.toDisplayBitmap(): Bitmap {
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    try {
        val row = IntArray(width)
        var offset = 0
        repeat(height) { y ->
            coroutineContext.ensureActive()
            for (x in 0 until width) {
                val red = rgb[offset++].toInt() and 255
                val green = rgb[offset++].toInt() and 255
                val blue = rgb[offset++].toInt() and 255
                row[x] = (255 shl 24) or (red shl 16) or (green shl 8) or blue
            }
            bitmap.setPixels(row, 0, width, 0, y, width, 1)
        }
        return bitmap
    } catch (failure: Throwable) {
        bitmap.recycle()
        throw failure
    }
}
