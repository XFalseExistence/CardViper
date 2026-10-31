package com.cardviper.app.vision.litert

import com.cardviper.app.vision.CardLabelCodec
import com.cardviper.app.vision.VisionImage
import org.junit.Assert.assertEquals
import org.junit.Test

class LiteRtSyntheticPipelineTest {
    @Test fun fixtureContractPreprocessingAndSyntheticClassifierOutputCompose() {
        val contract = classifierContract()
        val preprocessed = LiteRtImagePreprocessor.preprocess(
            VisionImage(2, 2, ByteArray(12) { it.toByte() }),
            contract,
        )
        assertEquals(2 * 2 * 3 * 4, preprocessed.tensorBytes.size)

        val scores = FloatArray(53) { 0f }
        scores[contract.labels.indexOf("KS")] = 0.99f
        val result = LiteRtClassifierPostprocessor.process(scores, contract)
        assertEquals(CardLabelCodec.decode("KS"), result.identity)
    }
}
