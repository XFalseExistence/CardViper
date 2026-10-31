package com.cardviper.app.vision.litert

import org.junit.Assert.*
import org.junit.Test

class LiteRtDetectorPostprocessorTest {
    @Test fun reconstructsYxyxNormalizedBoxesForLandscapeAndPortraitSources() {
        val contract = contract()
        val landscape = LiteRtDetectorPostprocessor.process(contract, tensors(), 8, 4)
        assertEquals(FloatBox(2f, 0f, 6f, 4f), landscape.single().box)
        val portraitTensors = mapOf(
            "boxes" to boxes(floatArrayOf(0.25f, 0.25f, 0.75f, 0.75f)),
            "scores" to scores(0.9f), "classes" to FloatArray(10), "count" to floatArrayOf(1f),
        )
        val portrait = LiteRtDetectorPostprocessor.process(contract, portraitTensors, 4, 8)
        assertEquals(FloatBox(0f, 2f, 4f, 6f), portrait.single().box)
    }

    @Test fun appliesThresholdCountAndSourceEdgeClamping() {
        val edge = tensors(box = floatArrayOf(-0.25f, -0.25f, 0.75f, 0.75f), score = 0.5f)
        assertEquals(FloatBox(0f, 0f, 6f, 4f), LiteRtDetectorPostprocessor.process(contract(), edge, 8, 4).single().box)
        assertTrue(LiteRtDetectorPostprocessor.process(contract(), tensors(score = 0.49f), 8, 4).isEmpty())
        assertTrue(LiteRtDetectorPostprocessor.process(contract(), tensors(count = 0f), 8, 4).isEmpty())
    }

    @Test fun rejectsInvalidClassMalformedLengthsAndCounts() {
        assertThrows(IllegalArgumentException::class.java) { LiteRtDetectorPostprocessor.process(contract(), tensors(clazz = 1f), 8, 4) }
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtDetectorPostprocessor.process(contract(), mapOf("boxes" to floatArrayOf(0f, 0f, 1f), "scores" to scores(0.9f), "classes" to FloatArray(10), "count" to floatArrayOf(1f)), 8, 4)
        }
        assertThrows(IllegalArgumentException::class.java) { LiteRtDetectorPostprocessor.process(contract(), tensors(count = 11f), 8, 4) }
        assertThrows(IllegalArgumentException::class.java) { LiteRtDetectorPostprocessor.process(contract(), emptyMap(), 8, 4) }
    }

    @Test fun dropsZeroAreaBoxesAndSupportsEmbeddedNmsWithoutApplyingHostNms() {
        val contract = contract()
        val zeroArea = tensors(box = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f))
        assertTrue(LiteRtDetectorPostprocessor.process(contract, zeroArea, 8, 4).isEmpty())

        val embedded = contract.copy(detector = contract.detector!!.copy(nms = contract.detector.nms.copy(ownership = NmsOwnership.EMBEDDED)))
        val overlap = mapOf(
            "boxes" to boxes(floatArrayOf(0.25f, 0.25f, 0.75f, 0.75f), floatArrayOf(0.26f, 0.26f, 0.74f, 0.74f)),
            "scores" to scores(0.9f, 0.8f), "classes" to FloatArray(10), "count" to floatArrayOf(2f),
        )
        assertEquals(2, LiteRtDetectorPostprocessor.process(embedded, overlap, 4, 4).size)
        assertEquals(1, LiteRtDetectorPostprocessor.process(contract, overlap, 4, 4).size)
    }

    @Test fun rejectsAmbiguousCountAndNonfiniteTensors() {
        assertThrows(IllegalArgumentException::class.java) { LiteRtDetectorPostprocessor.process(contract(), tensors(count = 1.5f), 8, 4) }
        assertThrows(IllegalArgumentException::class.java) { LiteRtDetectorPostprocessor.process(contract(), tensors(score = Float.NaN), 8, 4) }
    }

    @Test fun supportsPixelXYXYOutputsWithoutACountTensor() {
        val parsed = contract()
        val detector = parsed.detector!!.copy(
            coordinateOrder = BoxCoordinateOrder.XYXY,
            coordinateUnits = BoxCoordinateUnits.PIXELS,
            countTensor = null,
        )
        val noCount = parsed.copy(detector = detector, outputs = parsed.outputs.filterNot { it.role == OutputRole.DETECTION_COUNT })
        val rawBoxes = FloatArray(40).also { floatArrayOf(1f, 1f, 3f, 3f).copyInto(it) }
        val outputs = mapOf("boxes" to rawBoxes, "scores" to scores(0.9f), "classes" to FloatArray(10))
        assertEquals(FloatBox(2f, 0f, 6f, 4f), LiteRtDetectorPostprocessor.process(noCount, outputs, 8, 4).single().box)
    }

    private fun contract() = LiteRtModelContract.parse(javaClass.getResource("/vision/detector.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")!!.readText())
    private fun tensors(box: FloatArray = floatArrayOf(0.25f, 0.25f, 0.75f, 0.75f), score: Float = 0.9f, clazz: Float = 0f, count: Float = 1f) =
        mapOf("boxes" to boxes(box), "scores" to scores(score), "classes" to FloatArray(10) { clazz }, "count" to floatArrayOf(count))

    private fun boxes(vararg entries: FloatArray): FloatArray = FloatArray(40).also { destination ->
        entries.forEachIndexed { index, entry -> entry.copyInto(destination, destinationOffset = index * 4) }
    }
    private fun scores(vararg entries: Float): FloatArray = FloatArray(10).also { destination ->
        entries.forEachIndexed { index, value -> destination[index] = value }
    }
}
