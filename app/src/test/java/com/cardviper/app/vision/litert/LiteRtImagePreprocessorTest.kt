package com.cardviper.app.vision.litert

import com.cardviper.app.vision.VisionImage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LiteRtImagePreprocessorTest {
    @Test fun stretchFloat32PreservesRgbTensorOrder() {
        val contract = classifierContract()
        val image = VisionImage(
            2,
            2,
            byteArrayOf(
                1, 2, 3, 4, 5, 6,
                7, 8, 9, 10, 11, 12,
            ),
        )
        val output = LiteRtImagePreprocessor.preprocess(image, contract)
        assertArrayEquals(
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f),
            floats(output.tensorBytes),
            0f,
        )
        assertEquals(0, output.transform.padLeft)
        assertEquals(0, output.transform.padTop)
    }

    @Test fun channelOrderCanBeContractDefinedBgr() {
        val base = classifierContract()
        val contract = base.copy(
            input = base.input.copy(width = 1, height = 1),
            preprocessing = base.preprocessing.copy(channelOrder = ChannelOrder.BGR),
        )
        val image = VisionImage(1, 1, byteArrayOf(1, 2, 3))
        assertArrayEquals(
            floatArrayOf(3f, 2f, 1f),
            floats(LiteRtImagePreprocessor.preprocess(image, contract).tensorBytes),
            0f,
        )
    }

    @Test fun normalizationIsTakenFromContract() {
        val base = classifierContract()
        val contract = base.copy(
            input = base.input.copy(width = 1, height = 1),
            preprocessing = base.preprocessing.copy(
                normalization = NormalizationContract(multiplier = 0.5f, offset = -1f),
            ),
        )
        val values = floats(
            LiteRtImagePreprocessor.preprocess(
                VisionImage(1, 1, byteArrayOf(2, 4, 6)),
                contract,
            ).tensorBytes,
        )
        assertArrayEquals(floatArrayOf(0f, 1f, 2f), values, 0f)
    }

    @Test fun uint8PackingUsesRawUnsignedChannels() {
        val base = classifierContract()
        val contract = base.copy(
            input = base.input.copy(width = 1, height = 1, dtype = TensorDType.UINT8),
            preprocessing = base.preprocessing.copy(normalization = null),
        )
        val output = LiteRtImagePreprocessor.preprocess(
            VisionImage(1, 1, byteArrayOf(-1, 0, 127)),
            contract,
        )
        assertArrayEquals(byteArrayOf(-1, 0, 127), output.tensorBytes)
    }

    @Test fun letterboxLandscapeHasExactPaddingAndRoundTrips() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            8,
            4,
            4,
            4,
            ResizePolicy.LETTERBOX,
        )
        assertEquals(4, transform.contentWidth)
        assertEquals(2, transform.contentHeight)
        assertEquals(0, transform.padLeft)
        assertEquals(1, transform.padTop)
        val source = FloatRect(2f, 1f, 6f, 3f)
        assertRect(source, transform.modelToSource(transform.sourceToModel(source)))
    }

    @Test fun letterboxPortraitHasExactPaddingAndRoundTrips() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            4,
            8,
            4,
            4,
            ResizePolicy.LETTERBOX,
        )
        assertEquals(2, transform.contentWidth)
        assertEquals(4, transform.contentHeight)
        assertEquals(1, transform.padLeft)
        assertEquals(0, transform.padTop)
        val source = FloatRect(1f, 2f, 3f, 7f)
        assertRect(source, transform.modelToSource(transform.sourceToModel(source)))
    }

    @Test fun squareStretchRoundTrips() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            5,
            5,
            4,
            4,
            ResizePolicy.STRETCH,
        )
        val full = FloatRect(0f, 0f, 5f, 5f)
        assertRect(full, transform.modelToSource(transform.sourceToModel(full)))
    }

    @Test fun veryWideLetterboxStillKeepsNonzeroContent() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            100,
            1,
            4,
            4,
            ResizePolicy.LETTERBOX,
        )
        assertEquals(4, transform.contentWidth)
        assertEquals(1, transform.contentHeight)
        assertTrue(transform.padTop >= 0)
    }

    @Test fun singlePixelImageCanFillModelInput() {
        val contract = classifierContract()
        val image = VisionImage(1, 1, byteArrayOf(9, 8, 7))
        val values = floats(LiteRtImagePreprocessor.preprocess(image, contract).tensorBytes)
        for (index in values.indices step 3) {
            assertEquals(9f, values[index], 0f)
            assertEquals(8f, values[index + 1], 0f)
            assertEquals(7f, values[index + 2], 0f)
        }
    }

    @Test fun letterboxPaddingIsPackedThroughSameRawPixelPolicy() {
        val base = detectorEmbeddedContract()
        val contract = base.copy(
            input = base.input.copy(dtype = TensorDType.UINT8),
            preprocessing = base.preprocessing.copy(normalization = null, paddingValue = 17),
        )
        val output = LiteRtImagePreprocessor.preprocess(
            VisionImage(4, 2, ByteArray(4 * 2 * 3) { 99.toByte() }),
            contract,
        )
        assertArrayEquals(byteArrayOf(17, 17, 17), output.tensorBytes.copyOfRange(0, 3))
    }

    private fun floats(bytes: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / 4) { buffer.float }
    }

    private fun assertRect(expected: FloatRect, actual: FloatRect) {
        assertEquals(expected.left, actual.left, 0.0001f)
        assertEquals(expected.top, actual.top, 0.0001f)
        assertEquals(expected.right, actual.right, 0.0001f)
        assertEquals(expected.bottom, actual.bottom, 0.0001f)
    }
}
