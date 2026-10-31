package com.cardviper.app.vision.litert

import com.cardviper.app.vision.ClassifierInferenceException
import com.cardviper.app.vision.VisionImage
import com.cardviper.app.vision.VisionModelUnavailableException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class LiteRtCardRecognizerTest {
    @Test fun recognizesCanonicalIdentitiesIncludingBack() = runBlocking {
        val fixture = classifierBundleFixture()
        val labels = listOf("AC", "10H", "KS", "BACK")
        val runtime = FakeLiteRtRuntime(outputs = scores(fixture.contract, labels.first(), 0.8f))
        val recognizer = LiteRtCardRecognizer.fromBundle(fixture.bundle, runtime)
        labels.forEach { label ->
            runtime.outputs = scores(fixture.contract, label, 0.8f)
            assertEquals(label, com.cardviper.app.vision.CardLabelCodec.encode(recognizer.recognize(pixel()).identity))
        }
        assertEquals(4, runtime.inferenceCalls)
    }

    @Test fun returnsLowConfidenceBestIdentityAndStableAlternatives() = runBlocking {
        val fixture = classifierBundleFixture()
        val values = FloatArray(53).also { it[0] = 0.1f; it[1] = 0.1f; it[2] = 0.1f }
        val runtime = FakeLiteRtRuntime(outputs = mapOf("scores" to values))
        val result = LiteRtCardRecognizer(fixture.contract, fixture.modelBytes, fixture.labelsBytes!!, runtime).recognize(pixel())
        assertEquals("AC", com.cardviper.app.vision.CardLabelCodec.encode(result.identity))
        assertEquals(0.1f, result.confidence)
        assertEquals(listOf("AD", "AH", "AS", "2C"), result.alternatives.map { com.cardviper.app.vision.CardLabelCodec.encode(it.identity) })
    }

    @Test fun validatesHashesBeforeRuntimeOpenAndChecksActualTensorSignature() {
        val fixture = classifierBundleFixture()
        val badRuntime = FakeLiteRtRuntime()
        assertThrows(VisionModelUnavailableException::class.java) {
            LiteRtCardRecognizer(fixture.contract.copy(artifactSha256 = "0".repeat(64)), fixture.modelBytes, fixture.labelsBytes!!, badRuntime)
        }
        assertEquals(0, badRuntime.openCalls)
        assertThrows(VisionModelUnavailableException::class.java) {
            LiteRtCardRecognizer(fixture.contract, fixture.modelBytes, "wrong\nlabels\n".toByteArray(), badRuntime)
        }
        assertEquals(0, badRuntime.openCalls)

        val signature = expectedSignature(fixture.contract).copy(input = ObservedTensor("wrong", listOf(1, 2, 2, 3), TensorDataType.FLOAT32))
        val mismatch = FakeLiteRtRuntime(signatureOverride = signature)
        val initializationFailure = assertThrows(VisionModelUnavailableException::class.java) {
            LiteRtCardRecognizer(fixture.contract, fixture.modelBytes, fixture.labelsBytes!!, mismatch)
        }
        assertTrue(initializationFailure.message!!.contains("Classifier model initialization failed"))
        assertTrue(initializationFailure.cause is IllegalArgumentException)
        assertTrue(mismatch.closed)
    }

    @Test fun usesContractPreprocessingForTheRuntimeInput() = runBlocking {
        val fixture = classifierBundleFixture()
        val runtime = FakeLiteRtRuntime(outputs = scores(fixture.contract, "AC", 0.9f))
        LiteRtCardRecognizer(fixture.contract, fixture.modelBytes, fixture.labelsBytes!!, runtime).recognize(
            VisionImage(2, 2, byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)),
        )
        assertArrayEquals(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f), runtime.lastInput!!.floatValues!!, 0f)
    }

    @Test fun rejectsWrongOutputNameWidthAndNonfiniteValuesAsClassifierFailures() = runBlocking {
        val fixture = classifierBundleFixture()
        listOf(
            mapOf("wrong" to FloatArray(53)),
            mapOf("scores" to FloatArray(52)),
            mapOf("scores" to FloatArray(53).also { it[0] = Float.NaN }),
            mapOf("scores" to FloatArray(53).also { it[0] = Float.POSITIVE_INFINITY }),
        ).forEach { outputs ->
            val recognizer = LiteRtCardRecognizer(fixture.contract, fixture.modelBytes, fixture.labelsBytes!!, FakeLiteRtRuntime(outputs = outputs))
            assertThrows(ClassifierInferenceException::class.java) { runBlocking { recognizer.recognize(pixel()) } }
        }
    }

    @Test fun mapsRuntimeFailureButPreservesCancellation() = runBlocking {
        val fixture = classifierBundleFixture()
        val runtimeFailure = IllegalStateException("fake interpreter failure")
        val failed = LiteRtCardRecognizer(fixture.contract, fixture.modelBytes, fixture.labelsBytes!!,
            FakeLiteRtRuntime(failure = runtimeFailure))
        val wrapped = assertThrows(ClassifierInferenceException::class.java) { runBlocking { failed.recognize(pixel()) } }
        assertSame(runtimeFailure, wrapped.cause)

        val unavailable = VisionModelUnavailableException("classifier session unavailable")
        val unavailableRecognizer = LiteRtCardRecognizer(fixture.contract, fixture.modelBytes, fixture.labelsBytes,
            FakeLiteRtRuntime(failure = unavailable))
        assertSame(unavailable, assertThrows(VisionModelUnavailableException::class.java) {
            runBlocking { unavailableRecognizer.recognize(pixel()) }
        })

        val cancellation = CancellationException("cancelled")
        val cancelled = LiteRtCardRecognizer(fixture.contract, fixture.modelBytes, fixture.labelsBytes,
            FakeLiteRtRuntime(failure = cancellation))
        val observed = assertThrows(CancellationException::class.java) { runBlocking { cancelled.recognize(pixel()) } }
        assertSame(cancellation, observed)
    }

    @Test fun closeReleasesTheInjectedRuntimeSession() {
        val fixture = classifierBundleFixture()
        val runtime = FakeLiteRtRuntime()
        LiteRtCardRecognizer(fixture.contract, fixture.modelBytes, fixture.labelsBytes!!, runtime).close()
        assertTrue(runtime.closed)
    }

    private fun pixel() = VisionImage(1, 1, byteArrayOf(12, 34, 56))
    private fun scores(contract: LiteRtModelContract, label: String, confidence: Float): Map<String, FloatArray> =
        mapOf(contract.outputs.single().name to FloatArray(53).also { it[contract.labels.indexOf(label)] = confidence })
}
