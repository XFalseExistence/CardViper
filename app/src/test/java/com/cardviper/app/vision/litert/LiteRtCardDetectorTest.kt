package com.cardviper.app.vision.litert

import com.cardviper.app.vision.DetectorInferenceException
import com.cardviper.app.vision.VisionImage
import com.cardviper.app.vision.VisionModelUnavailableException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class LiteRtCardDetectorTest {
    @Test fun mapsOneMultipleAndEmptyResultsThroughContract() = runBlocking {
        val fixture = detectorBundleFixture()
        val values = outputs()
        values.getValue("count")[0] = 2f
        values.getValue("scores")[0] = 0.9f
        values.getValue("scores")[1] = 0.8f
        putYxyx(values.getValue("boxes"), 0, 0f, 0f, 0.5f, 0.5f)
        putYxyx(values.getValue("boxes"), 1, 0.5f, 0.5f, 1f, 1f)
        val runtime = FakeLiteRtRuntime(outputs = values)
        val detector = LiteRtCardDetector(fixture.contract, fixture.modelBytes, runtime)
        assertEquals(2, detector.detect(VisionImage(4, 4, ByteArray(48))).size)
        values.getValue("count")[0] = 0f
        assertTrue(detector.detect(VisionImage(4, 4, ByteArray(48))).isEmpty())
    }

    @Test fun reversesLetterboxForLandscapeAndPortraitAndClampsEdges() = runBlocking {
        val fixture = detectorBundleFixture()
        val values = outputs().also {
            it.getValue("count")[0] = 1f
            it.getValue("scores")[0] = 0.9f
        }
        val detector = LiteRtCardDetector(fixture.contract, fixture.modelBytes, FakeLiteRtRuntime(outputs = values))
        putYxyx(values.getValue("boxes"), 0, 0.25f, 0f, 0.75f, 1f)
        val landscape = detector.detect(VisionImage(8, 4, ByteArray(8 * 4 * 3))).single()
        assertEquals(0f, landscape.x, 0.001f); assertEquals(0f, landscape.y, 0.001f)
        assertEquals(8f, landscape.width, 0.001f); assertEquals(4f, landscape.height, 0.001f)

        putYxyx(values.getValue("boxes"), 0, 0f, 0.25f, 1f, 0.75f)
        val portrait = detector.detect(VisionImage(4, 8, ByteArray(4 * 8 * 3))).single()
        assertEquals(0f, portrait.x, 0.001f); assertEquals(0f, portrait.y, 0.001f)
        assertEquals(4f, portrait.width, 0.001f); assertEquals(8f, portrait.height, 0.001f)

        putYxyx(values.getValue("boxes"), 0, -0.25f, -0.25f, 1.25f, 1.25f)
        val clipped = detector.detect(VisionImage(4, 4, ByteArray(48))).single()
        assertEquals(0f, clipped.x); assertEquals(0f, clipped.y)
        assertEquals(4f, clipped.width); assertEquals(4f, clipped.height)
    }

    @Test fun sendsContractLetterboxPixelsToRuntime() = runBlocking {
        val fixture = detectorBundleFixture()
        val runtime = FakeLiteRtRuntime(outputs = outputs())
        LiteRtCardDetector(fixture.contract, fixture.modelBytes, runtime).detect(
            VisionImage(2, 4, ByteArray(2 * 4 * 3) { 50 }),
        )
        val expected = FloatArray(4 * 4 * 3)
        for (row in 0 until 4) {
            floatArrayOf(0f, 0f, 0f, 50f, 50f, 50f, 50f, 50f, 50f, 0f, 0f, 0f)
                .copyInto(expected, row * 12)
        }
        assertArrayEquals(expected, runtime.lastInput!!.floatValues!!, 0f)
    }

    @Test fun honorsThresholdEmbeddedNmsAndHostNms() = runBlocking {
        val hostFixture = detectorBundleFixture(NmsOwnership.HOST)
        val values = outputs().also {
            it.getValue("count")[0] = 2f
            it.getValue("scores")[0] = 0.9f
            it.getValue("scores")[1] = 0.8f
        }
        putYxyx(values.getValue("boxes"), 0, 0.1f, 0.1f, 0.8f, 0.8f)
        putYxyx(values.getValue("boxes"), 1, 0.15f, 0.15f, 0.75f, 0.75f)
        assertEquals(1, LiteRtCardDetector(hostFixture.contract, hostFixture.modelBytes, FakeLiteRtRuntime(outputs = values))
            .detect(VisionImage(4, 4, ByteArray(48))).size)

        val embeddedFixture = detectorBundleFixture(NmsOwnership.EMBEDDED)
        assertEquals(2, LiteRtCardDetector(embeddedFixture.contract, embeddedFixture.modelBytes, FakeLiteRtRuntime(outputs = values))
            .detect(VisionImage(4, 4, ByteArray(48))).size)
        values.getValue("scores")[0] = 0.49f
        values.getValue("scores")[1] = 0.49f
        assertTrue(LiteRtCardDetector(hostFixture.contract, hostFixture.modelBytes, FakeLiteRtRuntime(outputs = values))
            .detect(VisionImage(4, 4, ByteArray(48))).isEmpty())
    }

    @Test fun validatesArtifactSignatureAndOutputTensorsBeforeOrDuringUse(): Unit = runBlocking {
        val fixture = detectorBundleFixture()
        val badRuntime = FakeLiteRtRuntime()
        assertThrows(VisionModelUnavailableException::class.java) {
            LiteRtCardDetector(fixture.contract.copy(artifactSha256 = "0".repeat(64)), fixture.modelBytes, badRuntime)
        }
        assertEquals(0, badRuntime.openCalls)
        val signature = expectedSignature(fixture.contract).copy(outputs = emptyList())
        val mismatch = FakeLiteRtRuntime(signatureOverride = signature)
        assertThrows(VisionModelUnavailableException::class.java) { LiteRtCardDetector(fixture.contract, fixture.modelBytes, mismatch) }
        assertTrue(mismatch.closed)

        val runtime = FakeLiteRtRuntime(outputs = mapOf("boxes" to FloatArray(40)))
        val detector = LiteRtCardDetector(fixture.contract, fixture.modelBytes, runtime)
        assertThrows(DetectorInferenceException::class.java) { runBlocking { detector.detect(VisionImage(4, 4, ByteArray(48))) } }

        val missingCount = FakeLiteRtRuntime(outputs = outputs().filterKeys { it != "count" })
        assertThrows(DetectorInferenceException::class.java) {
            runBlocking { LiteRtCardDetector(fixture.contract, fixture.modelBytes, missingCount).detect(VisionImage(4, 4, ByteArray(48))) }
        }
    }

    @Test fun validatesClassIdsAndSupportsContractsWithoutCountTensor() = runBlocking {
        val withCount = detectorBundleFixture()
        val badClass = outputs().also { it.getValue("count")[0] = 1f; it.getValue("classes")[0] = 1f }
        assertThrows(DetectorInferenceException::class.java) {
            runBlocking { LiteRtCardDetector(withCount.contract, withCount.modelBytes, FakeLiteRtRuntime(outputs = badClass))
                .detect(VisionImage(4, 4, ByteArray(48))) }
        }

        val noCount = detectorBundleFixture(includeCount = false)
        val outputs = mapOf("boxes" to FloatArray(40), "scores" to FloatArray(10), "classes" to FloatArray(10))
        assertTrue(LiteRtCardDetector(noCount.contract, noCount.modelBytes, FakeLiteRtRuntime(outputs = outputs))
            .detect(VisionImage(4, 4, ByteArray(48))).isEmpty())
    }

    @Test fun mapsRuntimeFailureButPreservesCancellation() = runBlocking {
        val fixture = detectorBundleFixture()
        val failure = IllegalStateException("fake detector failure")
        val detector = LiteRtCardDetector(fixture.contract, fixture.modelBytes, FakeLiteRtRuntime(failure = failure))
        val wrapped = assertThrows(DetectorInferenceException::class.java) {
            runBlocking { detector.detect(VisionImage(4, 4, ByteArray(48))) }
        }
        assertSame(failure, wrapped.cause)

        val unavailable = com.cardviper.app.vision.VisionModelUnavailableException("detector session unavailable")
        val unavailableDetector = LiteRtCardDetector(fixture.contract, fixture.modelBytes,
            FakeLiteRtRuntime(failure = unavailable))
        assertSame(unavailable, assertThrows(com.cardviper.app.vision.VisionModelUnavailableException::class.java) {
            runBlocking { unavailableDetector.detect(VisionImage(4, 4, ByteArray(48))) }
        })

        val cancellation = CancellationException("cancelled")
        val cancelled = LiteRtCardDetector(fixture.contract, fixture.modelBytes, FakeLiteRtRuntime(failure = cancellation))
        val observed = assertThrows(CancellationException::class.java) {
            runBlocking { cancelled.detect(VisionImage(4, 4, ByteArray(48))) }
        }
        assertSame(cancellation, observed)
    }

    @Test fun closeReleasesTheInjectedRuntimeSession() {
        val fixture = detectorBundleFixture()
        val runtime = FakeLiteRtRuntime()
        LiteRtCardDetector(fixture.contract, fixture.modelBytes, runtime).close()
        assertTrue(runtime.closed)
    }

    private fun outputs() = mapOf(
        "boxes" to FloatArray(40), "scores" to FloatArray(10), "classes" to FloatArray(10), "count" to floatArrayOf(0f),
    )

    private fun putYxyx(boxes: FloatArray, index: Int, top: Float, left: Float, bottom: Float, right: Float) {
        floatArrayOf(top, left, bottom, right).copyInto(boxes, index * 4)
    }
}
