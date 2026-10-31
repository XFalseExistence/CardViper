package com.cardviper.app.vision.litert

import com.cardviper.app.vision.CardIdentity
import com.cardviper.app.vision.CardLabelCodec
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class LiteRtClassifierPostprocessorTest {
    private val contract = classifierContract()

    @Test fun mapsAcByExactOutputIndex() {
        val scores = FloatArray(53) { 0f }.also { it[0] = 0.9f }
        val result = LiteRtClassifierPostprocessor.process(scores, contract)
        assertEquals(CardLabelCodec.decode("AC"), result.identity)
        assertEquals(0.9f, result.confidence, 0f)
    }

    @Test fun mapsTenHByExactOutputIndex() {
        val index = contract.labels.indexOf("10H")
        val scores = FloatArray(53) { 0.01f }.also { it[index] = 0.95f }
        assertEquals(
            CardLabelCodec.decode("10H"),
            LiteRtClassifierPostprocessor.process(scores, contract).identity,
        )
    }

    @Test fun mapsKsByExactOutputIndex() {
        val index = contract.labels.indexOf("KS")
        val scores = FloatArray(53) { 0f }.also { it[index] = 1f }
        assertEquals(
            CardLabelCodec.decode("KS"),
            LiteRtClassifierPostprocessor.process(scores, contract).identity,
        )
    }

    @Test fun mapsBackAtIndex52() {
        val scores = FloatArray(53) { 0f }.also { it[52] = 1f }
        assertEquals(CardIdentity.Back, LiteRtClassifierPostprocessor.process(scores, contract).identity)
    }

    @Test fun tiesAreDeterministicByLowerOutputIndex() {
        val scores = FloatArray(53) { 0f }.also {
            it[4] = 0.8f
            it[5] = 0.8f
        }
        assertEquals(
            CardLabelCodec.decode(contract.labels[4]),
            LiteRtClassifierPostprocessor.process(scores, contract).identity,
        )
    }

    @Test fun rejectsNanAndInfinity() {
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtClassifierPostprocessor.process(
                FloatArray(53).also { it[0] = Float.NaN },
                contract,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtClassifierPostprocessor.process(
                FloatArray(53).also { it[0] = Float.POSITIVE_INFINITY },
                contract,
            )
        }
    }

    @Test fun rejectsWrongOutputWidth() {
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtClassifierPostprocessor.process(FloatArray(52), contract)
        }
    }

    @Test fun rejectsConfidenceOutsideUnitInterval() {
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtClassifierPostprocessor.process(
                FloatArray(53).also { it[0] = 1.01f },
                contract,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtClassifierPostprocessor.process(
                FloatArray(53).also { it[0] = -0.01f },
                contract,
            )
        }
    }

    @Test fun rejectsContractWhoseLabelOrderWasMutatedAfterParse() {
        val labels = contract.labels.toMutableList().also {
            val first = it[0]
            it[0] = it[1]
            it[1] = first
        }
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtClassifierPostprocessor.process(
                FloatArray(53),
                contract.copy(labels = labels),
            )
        }
    }
}
