package com.cardviper.app.vision.litert

import com.cardviper.app.vision.VisionImage
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt

data class FloatBox(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    init { require(listOf(left, top, right, bottom).all(Float::isFinite)) { "Box coordinates must be finite" } }
    val width: Float get() = right - left
    val height: Float get() = bottom - top
}

/** Continuous edge-coordinate transform. Its letterbox arithmetic mirrors ml/cardviper_ml/detector_geometry.py. */
data class LiteRtResizeTransform(
    val sourceWidth: Int,
    val sourceHeight: Int,
    val modelWidth: Int,
    val modelHeight: Int,
    val policy: ResizePolicy,
) {
    init {
        require(sourceWidth > 0 && sourceHeight > 0 && modelWidth > 0 && modelHeight > 0)
    }

    val scaleX: Double get() = if (policy == ResizePolicy.STRETCH) modelWidth.toDouble() / sourceWidth else scale
    val scaleY: Double get() = if (policy == ResizePolicy.STRETCH) modelHeight.toDouble() / sourceHeight else scale
    val scale: Double get() = min(modelWidth.toDouble() / sourceWidth, modelHeight.toDouble() / sourceHeight)
    val padX: Double get() = (modelWidth - sourceWidth * scaleX) / 2.0
    val padY: Double get() = (modelHeight - sourceHeight * scaleY) / 2.0

    fun sourceToModel(box: FloatBox): FloatBox {
        require(box.width > 0f && box.height > 0f) { "Source box must have positive area" }
        return FloatBox(
            (box.left * scaleX + padX).toFloat(), (box.top * scaleY + padY).toFloat(),
            (box.right * scaleX + padX).toFloat(), (box.bottom * scaleY + padY).toFloat(),
        )
    }

    fun modelToSource(box: FloatBox, clamp: Boolean = false): FloatBox {
        require(box.width > 0f && box.height > 0f) { "Model box must have positive area" }
        var left = ((box.left - padX) / scaleX).toFloat()
        var top = ((box.top - padY) / scaleY).toFloat()
        var right = ((box.right - padX) / scaleX).toFloat()
        var bottom = ((box.bottom - padY) / scaleY).toFloat()
        if (clamp) {
            left = left.coerceIn(0f, sourceWidth.toFloat())
            right = right.coerceIn(0f, sourceWidth.toFloat())
            top = top.coerceIn(0f, sourceHeight.toFloat())
            bottom = bottom.coerceIn(0f, sourceHeight.toFloat())
        }
        return FloatBox(left, top, right, bottom)
    }
}

data class PreprocessedImage(
    val shape: List<Int>,
    val dtype: TensorDataType,
    val floatValues: FloatArray?,
    val byteValues: ByteArray?,
    val transform: LiteRtResizeTransform,
)

/** Pure contract-driven RGB preprocessing; no model-specific normalization defaults are supplied. */
object LiteRtImagePreprocessor {
    fun preprocess(image: VisionImage, contract: LiteRtModelContract): PreprocessedImage {
        val width = contract.input.width
        val height = contract.input.height
        require(width.toLong() * height <= MAX_MODEL_PIXELS) { "Model input exceeds the supported preprocessing size" }
        val transform = LiteRtResizeTransform(image.width, image.height, width, height, contract.preprocessing.resizePolicy)
        val resizedWidth: Int
        val resizedHeight: Int
        if (contract.preprocessing.resizePolicy == ResizePolicy.STRETCH) {
            resizedWidth = width
            resizedHeight = height
        } else {
            resizedWidth = (image.width * transform.scale).roundToInt().coerceIn(1, width)
            resizedHeight = (image.height * transform.scale).roundToInt().coerceIn(1, height)
        }
        val left = (width - resizedWidth) / 2
        val top = (height - resizedHeight) / 2
        val channels = Array(height * width) { IntArray(3) { contract.preprocessing.padValue } }
        for (y in 0 until resizedHeight) {
            for (x in 0 until resizedWidth) {
                val pixel = sample(image, x, y, resizedWidth, resizedHeight, contract.preprocessing.resizeFilter)
                channels[(top + y) * width + left + x] = pixel
            }
        }

        val channelIndices = if (contract.preprocessing.channelOrder == ChannelOrder.RGB) intArrayOf(0, 1, 2) else intArrayOf(2, 1, 0)
        val shape = contract.input.shape
        val values = FloatArray(width * height * 3)
        for (y in 0 until height) for (x in 0 until width) for (outputChannel in 0..2) {
            val channel = channelIndices[outputChannel]
            val raw = channels[y * width + x][channel]
            val index = if (contract.input.layout == TensorLayout.NHWC) {
                (y * width + x) * 3 + outputChannel
            } else {
                outputChannel * width * height + y * width + x
            }
            val transformed = raw * contract.preprocessing.scale + contract.preprocessing.offset
            require(transformed.isFinite()) { "Preprocessing produced a nonfinite value" }
            val floatValue = transformed.toFloat()
            if (contract.input.dtype == TensorDataType.FLOAT32) {
                require(floatValue.isFinite()) { "Preprocessing value exceeds FLOAT32 range" }
            }
            values[index] = floatValue
        }

        return when (contract.input.dtype) {
            TensorDataType.FLOAT32 -> PreprocessedImage(shape, TensorDataType.FLOAT32, values, null, transform)
            TensorDataType.UINT8 -> {
                val packed = ByteArray(values.size) { index -> values[index].roundToInt().coerceIn(0, 255).toByte() }
                PreprocessedImage(shape, TensorDataType.UINT8, null, packed, transform)
            }
        }
    }

    private fun sample(image: VisionImage, x: Int, y: Int, width: Int, height: Int, filter: ResizeFilter): IntArray = when (filter) {
        ResizeFilter.NEAREST -> {
            val sx = floor((x + 0.5) * image.width / width).toInt().coerceIn(0, image.width - 1)
            val sy = floor((y + 0.5) * image.height / height).toInt().coerceIn(0, image.height - 1)
            rgbAt(image, sx, sy)
        }
        ResizeFilter.BILINEAR -> {
            val fx = ((x + 0.5) * image.width / width - 0.5).coerceIn(0.0, image.width - 1.0)
            val fy = ((y + 0.5) * image.height / height - 0.5).coerceIn(0.0, image.height - 1.0)
            val x0 = floor(fx).toInt()
            val y0 = floor(fy).toInt()
            val x1 = min(x0 + 1, image.width - 1)
            val y1 = min(y0 + 1, image.height - 1)
            val dx = fx - x0
            val dy = fy - y0
            IntArray(3) { channel ->
                val a = unsigned(image, x0, y0, channel) * (1.0 - dx) + unsigned(image, x1, y0, channel) * dx
                val b = unsigned(image, x0, y1, channel) * (1.0 - dx) + unsigned(image, x1, y1, channel) * dx
                (a * (1.0 - dy) + b * dy).roundToInt().coerceIn(0, 255)
            }
        }
    }

    private fun rgbAt(image: VisionImage, x: Int, y: Int) = IntArray(3) { channel -> unsigned(image, x, y, channel) }
    private fun unsigned(image: VisionImage, x: Int, y: Int, channel: Int): Int =
        image.rgb[(y * image.width + x) * 3 + channel].toInt() and 0xff

    private const val MAX_MODEL_PIXELS = 4_194_304L
}
