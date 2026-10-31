package com.cardviper.app.vision.litert

import com.cardviper.app.vision.VisionImage
import org.junit.Assert.assertEquals
import org.junit.Test

/** Synthetic contract → preprocessing → synthetic outputs → postprocessing; this is not model inference. */
class LiteRtSyntheticPipelineTest {
    @Test fun testOnlyContractsDrivePreprocessingAndSyntheticPostprocessing() {
        val classifier = LiteRtModelContract.parse(javaClass.getResource("/vision/classifier.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")!!.readText())
        val image = VisionImage(1, 1, byteArrayOf(20, 40, 60))
        assertEquals(12, LiteRtImagePreprocessor.preprocess(image, classifier).floatValues!!.size)
        val scores = FloatArray(53).also { it[52] = 0.75f }
        assertEquals("BACK", LiteRtClassifierPostprocessor.process(scores, classifier, classifier.labels).label)

        val detector = LiteRtModelContract.parse(javaClass.getResource("/vision/detector.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")!!.readText())
        val outputs = mapOf(
            "boxes" to FloatArray(40).also { floatArrayOf(0.25f, 0.25f, 0.75f, 0.75f).copyInto(it) },
            "scores" to FloatArray(10).also { it[0] = 0.9f }, "classes" to FloatArray(10), "count" to floatArrayOf(1f),
        )
        assertEquals(1, LiteRtDetectorPostprocessor.process(detector, outputs, 4, 4).size)
    }
}
