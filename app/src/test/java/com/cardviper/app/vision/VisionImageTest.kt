package com.cardviper.app.vision

import org.junit.Assert.*
import org.junit.Test

class VisionImageTest {
    @Test fun requiresPositiveDimensionsAndExactRgbLength() {
        listOf(0 to 1, 1 to 0, -1 to 1, 1 to -1).forEach { (width, height) ->
            assertThrows(IllegalArgumentException::class.java) { VisionImage(width, height, byteArrayOf()) }
        }
        listOf(0, 11, 13, 16).forEach { length ->
            assertThrows(IllegalArgumentException::class.java) { VisionImage(2, 2, ByteArray(length)) }
        }
        assertEquals(12, VisionImage(2, 2, ByteArray(12)).rgb.size)
    }

    @Test fun oversizedDimensionsCannotWrapIntoAnApparentlyValidBufferLength() {
        assertThrows(IllegalArgumentException::class.java) { VisionImage(65_536, 65_536, byteArrayOf()) }
        assertThrows(IllegalArgumentException::class.java) { VisionImage(Int.MAX_VALUE, Int.MAX_VALUE, byteArrayOf()) }
    }

    @Test fun fullImageCropPreservesPixelsWithoutAliasingSource() {
        val source = image()
        val crop = source.crop(PixelRect(0, 0, 4, 3))
        assertEquals(4, crop.width)
        assertEquals(3, crop.height)
        assertArrayEquals(source.rgb, crop.rgb)
        assertNotSame(source.rgb, crop.rgb)
        crop.rgb[0] = 100
        assertEquals(0.toByte(), source.rgb[0])
    }

    @Test fun cropUsesRgbCoordinatesWithoutRowBleed() {
        val crop = image().crop(PixelRect(left = 1, top = 1, right = 3, bottom = 3))
        assertEquals(2, crop.width)
        assertEquals(2, crop.height)
        assertArrayEquals(byteArrayOf(15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32), crop.rgb)
    }

    @Test fun topAndBottomEdgeCropsUseExclusiveBottomCoordinates() {
        assertArrayEquals(byteArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
            image().crop(PixelRect(0, 0, 4, 1)).rgb)
        assertArrayEquals(byteArrayOf(24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35),
            image().crop(PixelRect(0, 2, 4, 3)).rgb)
    }

    @Test fun leftAndRightEdgeCropsKeepEachRowsCorrectPixel() {
        assertArrayEquals(byteArrayOf(0, 1, 2, 12, 13, 14, 24, 25, 26),
            image().crop(PixelRect(0, 0, 1, 3)).rgb)
        assertArrayEquals(byteArrayOf(9, 10, 11, 21, 22, 23, 33, 34, 35),
            image().crop(PixelRect(3, 0, 4, 3)).rgb)
    }

    @Test fun singlePixelAtBottomRightIsAValidCrop() {
        val crop = image().crop(PixelRect(3, 2, 4, 3))
        assertEquals(1, crop.width)
        assertEquals(1, crop.height)
        assertArrayEquals(byteArrayOf(33, 34, 35), crop.rgb)
    }

    @Test fun outOfBoundsEmptyAndReversedRectanglesAreRejected() {
        listOf(
            PixelRect(-1, 0, 1, 1), PixelRect(0, -1, 1, 1),
            PixelRect(0, 0, 5, 1), PixelRect(0, 0, 1, 4),
            PixelRect(1, 1, 1, 2), PixelRect(1, 1, 2, 1),
            PixelRect(2, 1, 1, 2), PixelRect(1, 2, 2, 1),
            PixelRect(4, 0, 5, 1), PixelRect(0, 3, 1, 4),
            PixelRect(Int.MIN_VALUE, 0, Int.MAX_VALUE, 1),
        ).forEach { rect ->
            assertThrows("rect=$rect", IllegalArgumentException::class.java) { image().crop(rect) }
        }
    }

    private fun image() = VisionImage(4, 3, ByteArray(36) { it.toByte() })
}
