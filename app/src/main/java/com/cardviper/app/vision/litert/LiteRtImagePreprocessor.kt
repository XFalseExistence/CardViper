package com.cardviper.app.vision.litert

import com.cardviper.app.vision.VisionImage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class FloatRect(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    init {
        require(listOf(left, top, right, bottom).all { it.isFinite() }) { "Rectangle coordinates must be finite" }
    }
}

data class ModelTransform(
    val sourceWidth: Int,
    val sourceHeight: Int,
    val modelWidth: Int,
    val modelHeight: Int,
    val contentWidth: Int,
    val contentHeight: Int,
    val padLeft: Int,
    val padTop: Int,
) {
    init {
        require(sourceWidth > 0 && sourceHeight > 0 && modelWidth > 0 && modelHeight > 0)
        require(contentWidth > 0 && contentHeight > 0)
        require(padLeft >= 0 && padTop >= 0)
        require(padLeft + contentWidth <= modelWidth && padTop + contentHeight <= modelHeight)
    }

    val scaleX: Float = contentWidth.toFloat() / sourceWidth
    val scaleY: Float = contentHeight.toFloat() / sourceHeight

    fun sourceToModel(rect: FloatRect): FloatRect = FloatRect(
        left = rect.left * scaleX + padLeft,
        top = rect.top * scaleY + padTop,
        right = rect.right * scaleX + padLeft,
        bottom = rect.bottom * scaleY + padTop,
    )

    fun modelToSource(rect: FloatRect): FloatRect = FloatRect(
        left = (rect.left - padLeft) / scaleX,
        top = (rect.top - padTop) / scaleY,
        right = (rect.right - padLeft) / scaleX,
        bottom = (rect.bottom - padTop) / scaleY,
    )
}

data class PreprocessedImage(val tensorBytes: ByteArray, val transform: ModelTransform)

object LiteRtImagePreprocessor {
    fun preprocess(image: VisionImage, contract: LiteRtModelContract): PreprocessedImage {
        val input = contract.input
        val transform = buildTransform(image.width, image.height, input.width, input.height, contract.resize.policy)
        val channelBytes = ByteArray(input.width * input.height * input.channels) {
            contract.preprocessing.paddingValue.toByte()
        }
        for (modelY in 0 until transform.contentHeight) {
            for (modelX in 0 until transform.contentWidth) {
                val rgb = sample(
                    image = image,
                    targetX = modelX,
                    targetY = modelY,
                    targetWidth = transform.contentWidth,
                    targetHeight = transform.contentHeight,
                    mode = contract.preprocessing.resampling,
                )
                val outX = modelX + transform.padLeft
                val outY = modelY + transform.padTop
                val base = (outY * input.width + outX) * 3
                when (contract.preprocessing.channelOrder) {
                    ChannelOrder.RGB -> {
                        channelBytes[base] = rgb[0].toByte()
                        channelBytes[base + 1] = rgb[1].toByte()
                        channelBytes[base + 2] = rgb[2].toByte()
                    }
                    ChannelOrder.BGR -> {
                        channelBytes[base] = rgb[2].toByte()
                        channelBytes[base + 1] = rgb[1].toByte()
                        channelBytes[base + 2] = rgb[0].toByte()
                    }
                }
            }
        }
        return when (input.dtype) {
            TensorDType.UINT8 -> PreprocessedImage(channelBytes, transform)
            TensorDType.FLOAT32 -> {
                val normalization = requireNotNull(contract.preprocessing.normalization)
                val packed = ByteBuffer.allocate(channelBytes.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
                channelBytes.forEach { byte ->
                    val unsigned = byte.toInt() and 0xff
                    packed.putFloat(unsigned * normalization.multiplier + normalization.offset)
                }
                PreprocessedImage(packed.array(), transform)
            }
        }
    }

    fun buildTransform(
        sourceWidth: Int,
        sourceHeight: Int,
        modelWidth: Int,
        modelHeight: Int,
        policy: ResizePolicy,
    ): ModelTransform {
        require(sourceWidth > 0 && sourceHeight > 0 && modelWidth > 0 && modelHeight > 0)
        return when (policy) {
            ResizePolicy.STRETCH -> ModelTransform(
                sourceWidth, sourceHeight, modelWidth, modelHeight,
                contentWidth = modelWidth, contentHeight = modelHeight,
                padLeft = 0, padTop = 0,
            )
            ResizePolicy.LETTERBOX -> {
                val scale = min(
                    modelWidth.toDouble() / sourceWidth,
                    modelHeight.toDouble() / sourceHeight,
                )
                val contentWidth = max(1, (sourceWidth * scale).roundToInt()).coerceAtMost(modelWidth)
                val contentHeight = max(1, (sourceHeight * scale).roundToInt()).coerceAtMost(modelHeight)
                ModelTransform(
                    sourceWidth, sourceHeight, modelWidth, modelHeight,
                    contentWidth, contentHeight,
                    padLeft = (modelWidth - contentWidth) / 2,
                    padTop = (modelHeight - contentHeight) / 2,
                )
            }
        }
    }

    private fun sample(
        image: VisionImage,
        targetX: Int,
        targetY: Int,
        targetWidth: Int,
        targetHeight: Int,
        mode: ResamplingMode,
    ): IntArray = when (mode) {
        ResamplingMode.NEAREST -> {
            val sourceX = (((targetX + 0.5) * image.width / targetWidth).toInt())
                .coerceIn(0, image.width - 1)
            val sourceY = (((targetY + 0.5) * image.height / targetHeight).toInt())
                .coerceIn(0, image.height - 1)
            pixel(image, sourceX, sourceY)
        }
        ResamplingMode.BILINEAR -> {
            val sourceX = ((targetX + 0.5) * image.width / targetWidth - 0.5)
                .coerceIn(0.0, (image.width - 1).toDouble())
            val sourceY = ((targetY + 0.5) * image.height / targetHeight - 0.5)
                .coerceIn(0.0, (image.height - 1).toDouble())
            val x0 = floor(sourceX).toInt()
            val y0 = floor(sourceY).toInt()
            val x1 = (x0 + 1).coerceIn(0, image.width - 1)
            val y1 = (y0 + 1).coerceIn(0, image.height - 1)
            val wx = (sourceX - floor(sourceX)).toFloat().coerceIn(0f, 1f)
            val wy = (sourceY - floor(sourceY)).toFloat().coerceIn(0f, 1f)
            val p00 = pixel(image, x0, y0)
            val p10 = pixel(image, x1, y0)
            val p01 = pixel(image, x0, y1)
            val p11 = pixel(image, x1, y1)
            IntArray(3) { channel ->
                val top = p00[channel] * (1f - wx) + p10[channel] * wx
                val bottom = p01[channel] * (1f - wx) + p11[channel] * wx
                (top * (1f - wy) + bottom * wy).roundToInt().coerceIn(0, 255)
            }
        }
    }

    private fun pixel(image: VisionImage, x: Int, y: Int): IntArray {
        val base = (y * image.width + x) * 3
        return intArrayOf(
            image.rgb[base].toInt() and 0xff,
            image.rgb[base + 1].toInt() and 0xff,
            image.rgb[base + 2].toInt() and 0xff,
        )
    }
}
