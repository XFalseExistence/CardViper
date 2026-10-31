package com.cardviper.app.vision.litert

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.cardviper.app.vision.CardLabelCodec
import com.cardviper.app.vision.VisionImage
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertTrue
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

    @Test fun inMemoryBundleRunsThroughClassifierAdapterWithFakeRuntime() = runBlocking {
        val raw = asset("vision/classifier.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")
        val placeholder = LiteRtModelContract.parse(raw)
        val model = byteArrayOf(1, 2, 3, 4)
        val labels = (placeholder.labels.joinToString("\n") + "\n").toByteArray()
        val json = raw.replaceFirst(ZERO_HASH, LiteRtArtifactValidator.sha256(model))
            .replaceFirst(ZERO_HASH, LiteRtArtifactValidator.sha256(labels))
        val contract = LiteRtModelContract.parse(json)
        val bundle = LiteRtModelBundle(json, model, labels)
        val scores = FloatArray(53).also { it[contract.labels.indexOf("KS")] = 0.91f }
        val runtime = InstrumentedFakeRuntime(mapOf("scores" to scores))
        val recognizer = LiteRtCardRecognizer.fromLoader(InMemoryLiteRtModelBundle(bundle), runtime)

        val result = recognizer.recognize(VisionImage(1, 1, byteArrayOf(10, 20, 30)))
        assertEquals("KS", CardLabelCodec.encode(result.identity))
        assertEquals(0.91f, result.confidence)
        assertArrayEquals(floatArrayOf(10f, 20f, 30f, 10f, 20f, 30f, 10f, 20f, 30f, 10f, 20f, 30f),
            runtime.lastInput!!.floatValues!!, 0f)
        recognizer.close()
        assertTrue(runtime.closed)
    }

    @Test fun inMemoryBundleRunsThroughDetectorAdapterWithFakeRuntime() = runBlocking {
        val raw = asset("vision/detector.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")
        val model = byteArrayOf(5, 6, 7, 8)
        val json = raw.replaceFirst(ONE_HASH, LiteRtArtifactValidator.sha256(model))
        val contract = LiteRtModelContract.parse(json)
        val bundle = LiteRtModelBundle(json, model, null)
        val tensors = mapOf(
            "boxes" to FloatArray(40).also { floatArrayOf(0.25f, 0.25f, 0.75f, 0.75f).copyInto(it) },
            "scores" to FloatArray(10).also { it[0] = 0.9f },
            "classes" to FloatArray(10),
            "count" to floatArrayOf(1f),
        )
        val runtime = InstrumentedFakeRuntime(tensors)
        val detector = LiteRtCardDetector.fromLoader(InMemoryLiteRtModelBundle(bundle), runtime)

        val card = detector.detect(VisionImage(4, 4, ByteArray(48) { 40 })).single()
        assertEquals(1f, card.x)
        assertEquals(1f, card.y)
        assertEquals(2f, card.width)
        assertEquals(2f, card.height)
        assertEquals(0.9f, card.confidence)
        assertArrayEquals(FloatArray(48) { 40f }, runtime.lastInput!!.floatValues!!, 0f)
        detector.close()
        assertTrue(runtime.closed)
    }

    private fun asset(path: String) = InstrumentationRegistry.getInstrumentation().context.assets
        .open(path).bufferedReader().use { it.readText() }

    private class InstrumentedFakeRuntime(
        private val output: Map<String, FloatArray>,
    ) : LiteRtRuntime {
        var lastInput: PreprocessedImage? = null
        var closed = false

        override fun open(modelBytes: ByteArray, contract: LiteRtModelContract): LiteRtSession = object : LiteRtSession {
            override val signature = TensorSignature(
                ObservedTensor(contract.input.name, contract.input.shape, contract.input.dtype),
                contract.outputs.map { ObservedTensor(it.name, it.shape, it.dtype) },
            )
            override fun run(input: PreprocessedImage): Map<String, FloatArray> {
                lastInput = input
                return output
            }
            override fun close() { closed = true }
        }
    }

    private companion object {
        const val ZERO_HASH = "0000000000000000000000000000000000000000000000000000000000000000"
        const val ONE_HASH = "1111111111111111111111111111111111111111111111111111111111111111"
    }
}
