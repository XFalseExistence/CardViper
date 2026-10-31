package com.cardviper.app.vision.litert

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith

/** Repeats deterministic synthetic arithmetic to exercise test-only session lifecycle. */
@RunWith(AndroidJUnit4::class)
class StandaloneLiteRtRuntimeStressInstrumentedTest {
    @Test fun syntheticRuntimeIsDeterministicAcrossOneHundredInvocationsAndCloses() {
        val model = InstrumentationRegistry.getInstrumentation().context.assets
            .open(MODEL_ASSET).use { it.readBytes() }
        val runtime = StandaloneLiteRtRuntime(threads = 1)
        val session = runtime.openForRuntimeSmoke(model)
        val inputValues = floatArrayOf(0f, 0.5f, 1f, 2f, 10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f)
        val packed = PreprocessedImage(
            listOf(1, 2, 2, 3),
            TensorDataType.FLOAT32,
            inputValues,
            null,
            LiteRtResizeTransform(2, 2, 2, 2, ResizePolicy.STRETCH),
        )
        val outputName = session.signature.outputs.single().name

        repeat(INVOCATIONS) {
            assertArrayEquals(FloatArray(inputValues.size) { inputValues[it] * 2f }, session.run(packed).getValue(outputName), 0f)
        }

        session.close()
        assertThrows(IllegalStateException::class.java) { session.run(packed) }
    }

    private companion object {
        const val MODEL_ASSET = "vision/TEST_ONLY_SYNTHETIC_NOT_PRODUCTION.tflite"
        const val INVOCATIONS = 100
    }
}
