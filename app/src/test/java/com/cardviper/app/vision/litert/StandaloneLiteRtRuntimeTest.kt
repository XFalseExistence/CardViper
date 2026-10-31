package com.cardviper.app.vision.litert

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class StandaloneLiteRtRuntimeTest {
    @Test fun rejectsEmptyModelBytesBeforeNativeRuntimeInitialization() {
        val contract = classifierBundleFixture().contract
        val failure = assertThrows(IllegalArgumentException::class.java) {
            StandaloneLiteRtRuntime().open(ByteArray(0), contract)
        }
        assertEquals("Model artifact is empty", failure.message)
    }

    @Test fun rejectsNonPositiveThreadCount() {
        assertThrows(IllegalArgumentException::class.java) { StandaloneLiteRtRuntime(threads = 0) }
    }

    @Test fun encodesExactFloatInputInNativeOrderWithoutTruncationOrPadding() {
        val observed = ObservedTensor("input", listOf(1, 1, 2, 3), TensorDataType.FLOAT32)
        val values = floatArrayOf(1.25f, -2f, 3.5f, 4f, 5f, 6f)
        val packed = StandaloneLiteRtBufferCodec.encodeInput(preprocessed(values), observed)
        assertTrue(packed.isDirect)
        assertEquals(ByteOrder.nativeOrder(), packed.order())
        assertEquals(values.size * Float.SIZE_BYTES, packed.remaining())
        assertArrayEquals(values, FloatArray(values.size) { packed.getFloat() }, 0f)
    }

    @Test fun rejectsFloatInputLengthAndShapeMismatch() {
        val observed = ObservedTensor("input", listOf(1, 1, 2, 3), TensorDataType.FLOAT32)
        assertThrows(IllegalArgumentException::class.java) {
            StandaloneLiteRtBufferCodec.encodeInput(preprocessed(floatArrayOf(1f)), observed)
        }
        assertThrows(IllegalArgumentException::class.java) {
            StandaloneLiteRtBufferCodec.encodeInput(preprocessed(FloatArray(6), shape = listOf(1, 2, 1, 3)), observed)
        }
    }

    @Test fun decodesOutputIntoIndependentFloatArrayAndRejectsWrongByteLengthOrType() {
        val observed = ObservedTensor("output", listOf(1, 1, 2, 2), TensorDataType.FLOAT32)
        val source = floatArrayOf(2f, 4f, 6f, 8f)
        val native = ByteBuffer.allocateDirect(source.size * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())
        source.forEach(native::putFloat)
        val first = StandaloneLiteRtBufferCodec.decodeOutput(native, observed)
        first[0] = -100f
        val second = StandaloneLiteRtBufferCodec.decodeOutput(native, observed)
        assertArrayEquals(source, second, 0f)
        assertThrows(IllegalArgumentException::class.java) {
            StandaloneLiteRtBufferCodec.decodeOutput(ByteBuffer.allocateDirect(4), observed)
        }
        assertThrows(IllegalArgumentException::class.java) {
            StandaloneLiteRtBufferCodec.decodeOutput(native, observed.copy(dtype = TensorDataType.UINT8))
        }
    }

    private fun preprocessed(
        values: FloatArray,
        shape: List<Int> = listOf(1, 1, 2, 3),
    ) = PreprocessedImage(
        shape = shape,
        dtype = TensorDataType.FLOAT32,
        floatValues = values,
        byteValues = null,
        transform = LiteRtResizeTransform(2, 1, 2, 1, ResizePolicy.STRETCH),
    )
}
