package com.cardviper.app.vision.litert

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.cardviper.app.vision.VisionImage
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/** Runs only synthetic contract/pixel/output data. No TFLite model is loaded or inferred. */
@RunWith(AndroidJUnit4::class)
class LiteRtSyntheticPipelineInstrumentedTest {
    @Test fun testOnlyContractPreprocessesAndPostprocessesSyntheticCardOutput() {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        val json = assets.open("vision/classifier.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json").bufferedReader().use { it.readText() }
        val contract = LiteRtModelContract.parse(json)
        val packed = LiteRtImagePreprocessor.preprocess(VisionImage(1, 1, byteArrayOf(10, 20, 30)), contract)
        assertEquals(12, packed.floatValues!!.size)
        val scores = FloatArray(53).also { it[52] = 0.99f }
        assertEquals("BACK", LiteRtClassifierPostprocessor.process(scores, contract, contract.labels).label)
    }
}
