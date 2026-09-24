package com.cardviper.app.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ColorSpace
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import androidx.annotation.RequiresApi
import java.io.FileNotFoundException
import java.io.InputStream
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext

class AndroidImageSourceLoader(context: Context, private val maxLongEdge: Int = 4096) : ImageSourceLoader {
    private val resolver = context.applicationContext.contentResolver

    init {
        require(maxLongEdge in 1..4096) { "Decode long edge must be between 1 and 4096" }
    }

    override suspend fun load(source: String): VisionImage = withContext(Dispatchers.IO) {
        val uri = Uri.parse(source)
        if (uri.scheme != "content" && uri.scheme != "file") {
            throw ImageLoadException("Unsupported image URI")
        }
        try {
            coroutineContext.ensureActive()
            val bitmap = if (Build.VERSION.SDK_INT >= 28) decodeModern(uri) else decodeLegacy(uri)
            try {
                toRgb(bitmap)
            } finally {
                bitmap.recycle()
            }
        } catch (failure: CancellationException) {
            throw failure
        } catch (failure: ImageLoadException) {
            throw failure
        } catch (failure: ImageDecodeException) {
            throw failure
        } catch (failure: FileNotFoundException) {
            throw ImageLoadException("Image URI is unavailable", failure)
        } catch (failure: SecurityException) {
            throw ImageLoadException("Image access was revoked", failure)
        } catch (failure: Exception) {
            throw ImageDecodeException("Image could not be decoded", failure)
        } catch (failure: OutOfMemoryError) {
            throw ImageDecodeException("Insufficient memory to decode image", failure)
        }
    }

    private fun open(uri: Uri): InputStream = try {
        resolver.openInputStream(uri) ?: throw ImageLoadException("Image URI could not be opened")
    } catch (failure: ImageLoadException) {
        throw failure
    } catch (failure: Exception) {
        throw ImageLoadException("Image URI could not be opened", failure)
    }

    @RequiresApi(28)
    private fun decodeModern(uri: Uri): Bitmap {
        // Separate access failures from malformed image data before the decoder opens its source.
        open(uri).use { }
        return try {
            ImageDecoder.decodeBitmap(ImageDecoder.createSource(resolver, uri)) { decoder, info, _ ->
                val size = boundedDecodeSize(info.size.width, info.size.height, maxLongEdge)
                decoder.setTargetSize(size.width, size.height)
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB))
            }
        } catch (failure: ImageDecoder.DecodeException) {
            if (failure.error == ImageDecoder.DecodeException.SOURCE_EXCEPTION) {
                throw ImageLoadException("Image source could not be read", failure)
            }
            throw ImageDecodeException("Image data is invalid or incomplete", failure)
        }
    }

    private fun decodeLegacy(uri: Uri): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        open(uri).use { BitmapFactory.decodeStream(it, null, bounds) }
        val target = boundedDecodeSize(bounds.outWidth, bounds.outHeight, maxLongEdge)
        // Round source/sample upward so even odd dimensions stay within the allocation budget.
        val options = BitmapFactory.Options().apply {
            inSampleSize = legacyDecodeSampleSize(bounds.outWidth, bounds.outHeight, maxLongEdge)
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inPreferredColorSpace = ColorSpace.get(ColorSpace.Named.SRGB)
        }
        val decoded = open(uri).use { BitmapFactory.decodeStream(it, null, options) }
            ?: throw ImageDecodeException("Image decoder returned no pixels")
        // Never upscale a sampled bitmap; keep the classifier and future overlay on this geometry.
        val size = boundedDecodeSize(decoded.width, decoded.height, maxOf(target.width, target.height))
        if (size.width == decoded.width && size.height == decoded.height) return decoded
        return try {
            Bitmap.createScaledBitmap(decoded, size.width, size.height, true)
        } finally {
            decoded.recycle()
        }
    }

    private suspend fun toRgb(bitmap: Bitmap): VisionImage {
        val width = bitmap.width
        val height = bitmap.height
        check(width in 1..maxLongEdge && height in 1..maxLongEdge)
        val rgb = ByteArray(width * height * 3)
        val row = IntArray(width)
        var offset = 0
        repeat(height) { y ->
            coroutineContext.ensureActive()
            bitmap.getPixels(row, 0, width, 0, y, width, 1)
            for (pixel in row) {
                // Composite transparent pixels onto black for a deterministic opaque RGB image.
                val alpha = pixel ushr 24
                rgb[offset++] = (((pixel ushr 16 and 255) * alpha) / 255).toByte()
                rgb[offset++] = (((pixel ushr 8 and 255) * alpha) / 255).toByte()
                rgb[offset++] = (((pixel and 255) * alpha) / 255).toByte()
            }
        }
        return VisionImage(width, height, rgb)
    }
}
