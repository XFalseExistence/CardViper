package com.cardviper.app.vision

import org.junit.Assert.*
import org.junit.Test

class ImageDecodeSizingTest {
    @Test fun landscape4096CapPreservesAspectRatio() {
        assertEquals(DecodeSize(4096, 2048), boundedDecodeSize(8000, 4000, 4096))
    }
    @Test fun portrait4096CapPreservesAspectRatio() {
        assertEquals(DecodeSize(2048, 4096), boundedDecodeSize(4000, 8000, 4096))
    }
    @Test fun smallImageIsNotUpscaled() {
        assertEquals(DecodeSize(1200, 800), boundedDecodeSize(1200, 800, 4096))
        assertEquals(DecodeSize(1, 1), boundedDecodeSize(1, 1, 4096))
        assertEquals(DecodeSize(4096, 4096), boundedDecodeSize(4096, 4096, 4096))
    }
    @Test fun unusualAspectRatiosStayPositiveAndBounded() {
        assertEquals(DecodeSize(4096, 1), boundedDecodeSize(Int.MAX_VALUE, 1, 4096))
        assertEquals(DecodeSize(1, 4096), boundedDecodeSize(1, Int.MAX_VALUE, 4096))
        assertEquals(DecodeSize(4096, 4096), boundedDecodeSize(Int.MAX_VALUE, Int.MAX_VALUE, 4096))
        assertEquals(DecodeSize(4096, 2731), boundedDecodeSize(6000, 4000, 4096))
    }
    @Test fun acceptsSmallerDecodeBudget() {
        assertEquals(DecodeSize(100, 50), boundedDecodeSize(8000, 4000, 100))
    }
    @Test fun legacySamplingBoundsAllocationsWithoutOverflow() {
        assertEquals(1, legacyDecodeSampleSize(1200, 800, 4096))
        assertEquals(2, legacyDecodeSampleSize(8000, 4000, 4096))
        assertEquals(4, legacyDecodeSampleSize(8193, 4000, 4096))
        assertEquals(4, legacyDecodeSampleSize(4000, 8193, 4096))
        assertEquals(1 shl 30, legacyDecodeSampleSize(Int.MAX_VALUE, Int.MAX_VALUE, 1))
    }
    @Test fun rejectsInvalidDimensionsAndBudget() {
        for (invalid in listOf(0, -1, Int.MIN_VALUE)) {
            assertThrows(IllegalArgumentException::class.java) { boundedDecodeSize(invalid, 4000, 4096) }
            assertThrows(IllegalArgumentException::class.java) { boundedDecodeSize(8000, invalid, 4096) }
            assertThrows(IllegalArgumentException::class.java) { boundedDecodeSize(8000, 4000, invalid) }
        }
    }
}
