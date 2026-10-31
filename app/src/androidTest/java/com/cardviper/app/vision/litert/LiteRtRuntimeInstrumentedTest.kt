package com.cardviper.app.vision.litert

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith

/** Executes only a tiny arithmetic TEST_ONLY model through the real standalone LiteRT runtime. */
@RunWith(AndroidJUnit4::class)
class LiteRtRuntimeInstrumentedTest {
    @Test fun syntheticArithmeticModelExposesInspectedTensorsAndExecutesExactly() {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        val model = assets.open(MODEL_ASSET).use { it.readBytes() }
        val metadata = assets.open(METADATA_ASSET).bufferedReader().use { JSONObject(it.readText()) }
        assertEquals(metadata.getString("model_sha256"), LiteRtArtifactValidator.sha256(model))

        val runtime = StandaloneLiteRtRuntime(threads = 1)
        val session = runtime.openForRuntimeSmoke(model)
        val expectedSignature = metadata.getJSONObject("signature")
        assertTensorMatches(expectedSignature.getJSONObject("input"), session.signature.input)
        val expectedOutputs = expectedSignature.getJSONArray("outputs")
        assertEquals(expectedOutputs.length(), session.signature.outputs.size)
        session.signature.outputs.forEachIndexed { index, tensor ->
            assertTensorMatches(expectedOutputs.getJSONObject(index), tensor)
        }

        val input = floatArrayOf(0f, 0.5f, 1f, 2f, 10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f)
        val transformed = LiteRtResizeTransform(2, 2, 2, 2, ResizePolicy.STRETCH)
        val packed = PreprocessedImage(listOf(1, 2, 2, 3), TensorDataType.FLOAT32, input, null, transformed)
        val output = session.run(packed)
        val outputName = metadata.getJSONObject("signature").getJSONArray("outputs").getJSONObject(0).getString("name")
        assertArrayEquals(FloatArray(input.size) { input[it] * 2f }, output.getValue(outputName), 0f)

        session.close()
        assertThrows(IllegalStateException::class.java) { session.run(packed) }
    }

    private fun assertTensorMatches(expected: JSONObject, actual: ObservedTensor) {
        assertEquals(expected.getString("name"), actual.name)
        assertEquals(expected.getString("dtype"), actual.dtype.name)
        val shape = expected.getJSONArray("shape")
        assertEquals(shape.length(), actual.shape.size)
        actual.shape.forEachIndexed { index, dimension -> assertEquals(shape.getInt(index), dimension) }
    }

    private companion object {
        const val MODEL_ASSET = "vision/litert_runtime_smoke.TEST_ONLY.tflite"
        const val METADATA_ASSET = "vision/litert_runtime_smoke.TEST_ONLY.json"
    }
}
