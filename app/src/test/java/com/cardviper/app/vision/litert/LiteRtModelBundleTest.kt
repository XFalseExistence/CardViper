package com.cardviper.app.vision.litert

import com.cardviper.app.vision.VisionModelUnavailableException
import org.junit.Assert.*
import org.junit.Test

class LiteRtModelBundleTest {
    @Test fun inMemoryBundleReturnsDefensiveByteCopies() {
        val model = byteArrayOf(1, 2, 3)
        val labels = byteArrayOf(4, 5)
        val loader = InMemoryLiteRtModelBundle(LiteRtModelBundle("{}", model, labels))
        model[0] = 9
        labels[0] = 9

        val first = loader.load()
        assertArrayEquals(byteArrayOf(1, 2, 3), first.modelBytes)
        assertArrayEquals(byteArrayOf(4, 5), first.labelsBytes)
        val exposed = first.modelBytes
        exposed[0] = 8
        assertArrayEquals(byteArrayOf(1, 2, 3), loader.load().modelBytes)
    }

    @Test fun missingBundleRemainsModelUnavailable() {
        val unavailable = VisionModelUnavailableException("test-only bundle absent")
        val failure = assertThrows(VisionModelUnavailableException::class.java) {
            LiteRtCardRecognizer.fromLoader(LiteRtModelBundleLoader { throw unavailable }, LiteRtRuntime { _, _ -> error("must not load") })
        }
        assertSame(unavailable, failure)
    }

    @Test fun loaderIoFailureIsMappedToModelUnavailable() {
        val ioFailure = java.io.IOException("test-only missing asset")
        val failure = assertThrows(VisionModelUnavailableException::class.java) {
            LiteRtCardRecognizer.fromLoader(LiteRtModelBundleLoader { throw ioFailure }, LiteRtRuntime { _, _ -> error("must not load") })
        }
        assertSame(ioFailure, failure.cause)
    }

    @Test fun malformedBundleContractFailsAsModelUnavailableBeforeRuntimeOpen() {
        val runtime = FakeLiteRtRuntime()
        val failure = assertThrows(VisionModelUnavailableException::class.java) {
            LiteRtCardDetector.fromBundle(LiteRtModelBundle("not json", byteArrayOf(1), null), runtime)
        }
        assertTrue(failure.message!!.contains("DETECTOR model bundle contract is invalid"))
        assertEquals(0, runtime.openCalls)
    }

    @Test fun detectorBundleRejectsClassifierLabelsInsteadOfIgnoringThem() {
        val fixture = detectorBundleFixture()
        val runtime = FakeLiteRtRuntime()
        val bundle = LiteRtModelBundle(fixture.bundle.contractJson, fixture.modelBytes, byteArrayOf(1))
        val failure = assertThrows(VisionModelUnavailableException::class.java) {
            LiteRtCardDetector.fromBundle(bundle, runtime)
        }
        assertTrue(failure.message!!.contains("DETECTOR model bundle contract is invalid"))
        assertEquals(0, runtime.openCalls)
    }
}
