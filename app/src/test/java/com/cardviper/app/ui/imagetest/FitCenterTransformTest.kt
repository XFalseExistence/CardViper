package com.cardviper.app.ui.imagetest

import org.junit.Assert.*
import org.junit.Test

class FitCenterTransformTest {
    @Test fun wideImageLetterboxesVertically() {
        val fit = FitCenterTransform.create(1000f, 500f, 500f, 500f)
        assertEquals(0.5f, fit.scale, 0.0001f)
        assertEquals(500f, fit.drawnWidth, 0.0001f)
        assertEquals(250f, fit.drawnHeight, 0.0001f)
        assertEquals(0f, fit.offsetX, 0.0001f)
        assertEquals(125f, fit.offsetY, 0.0001f)
    }
    @Test fun portraitImageLetterboxesHorizontally() {
        val fit = FitCenterTransform.create(500f, 1000f, 500f, 500f)
        assertEquals(0.5f, fit.scale, 0.0001f)
        assertEquals(250f, fit.drawnWidth, 0.0001f)
        assertEquals(500f, fit.drawnHeight, 0.0001f)
        assertEquals(125f, fit.offsetX, 0.0001f)
        assertEquals(0f, fit.offsetY, 0.0001f)
    }
    @Test fun equalAspectRatioFillsViewport() {
        val fit = FitCenterTransform.create(1000f, 500f, 400f, 200f)
        assertEquals(0.4f, fit.scale, 0.0001f)
        assertEquals(400f, fit.drawnWidth, 0.0001f)
        assertEquals(200f, fit.drawnHeight, 0.0001f)
        assertEquals(0f, fit.offsetX, 0.0001f)
        assertEquals(0f, fit.offsetY, 0.0001f)
    }
    @Test fun rightEdgeMapsToImageRightEdge() {
        val fit = FitCenterTransform.create(500f, 1000f, 500f, 500f)
        assertEquals(325f, fit.mapX(400f), 0.0001f)
        assertEquals(375f, fit.mapX(500f), 0.0001f)
    }
    @Test fun bottomEdgeMapsToImageBottomEdge() {
        val fit = FitCenterTransform.create(1000f, 500f, 500f, 500f)
        assertEquals(325f, fit.mapY(400f), 0.0001f)
        assertEquals(375f, fit.mapY(500f), 0.0001f)
    }
    @Test fun rejectsNonPositiveOrNonFiniteDimensions() {
        for (bad in listOf(0f, -1f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)) {
            assertThrows(IllegalArgumentException::class.java) { FitCenterTransform.create(bad, 500f, 500f, 500f) }
            assertThrows(IllegalArgumentException::class.java) { FitCenterTransform.create(500f, bad, 500f, 500f) }
            assertThrows(IllegalArgumentException::class.java) { FitCenterTransform.create(500f, 500f, bad, 500f) }
            assertThrows(IllegalArgumentException::class.java) { FitCenterTransform.create(500f, 500f, 500f, bad) }
        }
    }
}
