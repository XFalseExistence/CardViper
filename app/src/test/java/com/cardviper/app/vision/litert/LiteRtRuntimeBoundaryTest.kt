package com.cardviper.app.vision.litert

import org.junit.Assert.*
import org.junit.Test

class LiteRtRuntimeBoundaryTest {
    @Test fun validatesHashesBeforeOpenAndClosesIncompatibleSession() {
        val labels = (canonicalLabels().joinToString("\n") + "\n").toByteArray()
        val model = "synthetic fixture bytes".toByteArray()
        val parsed = LiteRtModelContract.parse(javaClass.getResource("/vision/classifier.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")!!.readText())
        val contract = parsed.copy(
            artifactSha256 = LiteRtArtifactValidator.sha256(model),
            labelsSha256 = LiteRtArtifactValidator.sha256(labels),
        )
        val matching = signature(contract)
        var openCalls = 0
        val validSession = TestSession(matching)
        val factory = LiteRtRuntimeFactory { _, _ -> openCalls++; validSession }
        assertSame(validSession, LiteRtRuntimeBoundary.openValidated(factory, model, labels, contract))
        assertEquals(1, openCalls)

        val wrongHash = contract.copy(artifactSha256 = "0".repeat(64))
        assertThrows(IllegalArgumentException::class.java) { LiteRtRuntimeBoundary.openValidated(factory, model, labels, wrongHash) }
        assertEquals(1, openCalls)

        val mismatch = TestSession(matching.copy(input = matching.input.copy(shape = listOf(1, 4, 2, 3))))
        val badFactory = LiteRtRuntimeFactory { _, _ -> mismatch }
        assertThrows(IllegalArgumentException::class.java) { LiteRtRuntimeBoundary.openValidated(badFactory, model, labels, contract) }
        assertTrue(mismatch.closed)
    }

    private class TestSession(override val signature: TensorSignature) : LiteRtRuntimeSession {
        var closed = false
        override fun run(input: PreprocessedImage): Map<String, FloatArray> = error("Smoke boundary must not infer")
        override fun close() { closed = true }
    }

    private fun canonicalLabels() = listOf("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
        .flatMap { rank -> listOf("C", "D", "H", "S").map { "$rank$it" } } + "BACK"
    private fun signature(contract: LiteRtModelContract) = TensorSignature(
        ObservedTensor(contract.input.name, contract.input.shape, contract.input.dtype),
        contract.outputs.map { ObservedTensor(it.name, it.shape, it.dtype) },
    )
}
