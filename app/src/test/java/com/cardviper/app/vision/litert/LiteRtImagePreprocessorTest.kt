package com.cardviper.app.vision.litert

import com.cardviper.app.vision.VisionImage
import org.junit.Assert.*
import org.junit.Test

class LiteRtImagePreprocessorTest {
    @Test fun letterboxesLandscapePortraitSquareAndVeryWideInputsDeterministically() {
        val contract = detectorContract()
        val cases = listOf(
            Triple(VisionImage(8, 4, ByteArray(8 * 4 * 3)), 0.5f, 0.5f),
            Triple(VisionImage(4, 8, ByteArray(4 * 8 * 3)), 0.5f, 0.5f),
            Triple(VisionImage(4, 4, ByteArray(4 * 4 * 3)), 1f, 1f),
            Triple(VisionImage(16, 1, ByteArray(16 * 3)), 0.25f, 0.25f),
        )
        cases.forEach { (source, scaleX, scaleY) ->
            val result = LiteRtImagePreprocessor.preprocess(source, contract)
            assertEquals(scaleX, result.transform.scaleX.toFloat(), 0.0001f)
            assertEquals(scaleY, result.transform.scaleY.toFloat(), 0.0001f)
            assertEquals(4 * 4 * 3, result.floatValues!!.size)
        }
    }

    @Test fun stretchPreservesRgbFloatOrderingAndAppliesContractNormalization() {
        val contract = classifierContract().copy(preprocessing = classifierContract().preprocessing.copy(scale = 0.5, offset = -1.0))
        val source = VisionImage(2, 1, byteArrayOf(10, 20, 30, 40, 50, 60))
        val result = LiteRtImagePreprocessor.preprocess(source, contract)
        assertArrayEquals(floatArrayOf(4f, 9f, 14f, 19f, 24f, 29f, 4f, 9f, 14f, 19f, 24f, 29f), result.floatValues!!, 0.0001f)
        assertEquals(listOf(1, 2, 2, 3), result.shape)
    }

    @Test fun letterboxPaddingUsesExactCenteredGeometryAndContractPadValue() {
        val detector = detectorContract()
        val source = VisionImage(4, 2, ByteArray(4 * 2 * 3) { 100.toByte() })
        val result = LiteRtImagePreprocessor.preprocess(source, detector)
        assertEquals(1.0, result.transform.padY, 0.0001)
        val values = result.floatValues!!
        assertEquals(0f, values[0], 0f)
        assertEquals(100f, values[12], 0f)
        assertEquals(0f, values[36], 0f)
    }

    @Test fun supportsUint8NchwBgrPackingForSinglePixelAndEdgeCase() {
        val base = classifierContract()
        val contract = base.copy(
            input = base.input.copy(shape = listOf(1, 3, 1, 1), layout = TensorLayout.NCHW, dtype = TensorDataType.UINT8),
            preprocessing = base.preprocessing.copy(channelOrder = ChannelOrder.BGR, scale = 2.0, offset = 1.0),
        )
        val result = LiteRtImagePreprocessor.preprocess(VisionImage(1, 1, byteArrayOf(10, 20, 30)), contract)
        assertArrayEquals(byteArrayOf(61, 41, 21), result.byteValues!!)
        assertEquals(listOf(1, 3, 1, 1), result.shape)
    }

    @Test fun bilinearResizeUsesContractFilterWithDeterministicCenterSampling() {
        val base = classifierContract()
        val contract = base.copy(
            input = base.input.copy(shape = listOf(1, 2, 4, 3)),
            preprocessing = base.preprocessing.copy(resizeFilter = ResizeFilter.BILINEAR),
        )
        val source = VisionImage(2, 1, byteArrayOf(0, 0, 0, 100, 100, 100))
        val output = LiteRtImagePreprocessor.preprocess(source, contract).floatValues!!
        assertArrayEquals(floatArrayOf(0f, 25f, 75f, 100f), floatArrayOf(output[0], output[3], output[6], output[9]), 0.001f)
    }

    @Test fun rejectsFloat32OverflowInsteadOfPackingInfinity() {
        val base = classifierContract()
        val contract = base.copy(preprocessing = base.preprocessing.copy(scale = 1.0e100, offset = 0.0))
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtImagePreprocessor.preprocess(VisionImage(1, 1, byteArrayOf(255.toByte(), 255.toByte(), 255.toByte())), contract)
        }
    }

    @Test fun transformRoundTripsBoxesAndClampsEdges() {
        val result = LiteRtImagePreprocessor.preprocess(VisionImage(8, 4, ByteArray(96)), detectorContract())
        val source = FloatBox(0f, 0f, 8f, 4f)
        val model = result.transform.sourceToModel(source)
        assertEquals(FloatBox(0f, 1f, 4f, 3f), model)
        assertEquals(source, result.transform.modelToSource(model))
        assertEquals(FloatBox(0f, 0f, 8f, 4f), result.transform.modelToSource(FloatBox(-1f, 0f, 5f, 4f), clamp = true))
    }

    private fun classifierContract() = LiteRtModelContract.parse(fixture("classifier"))
    private fun detectorContract() = LiteRtModelContract.parse(fixture("detector"))
    private fun fixture(name: String) = javaClass.getResource("/vision/$name.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")!!.readText()
}
