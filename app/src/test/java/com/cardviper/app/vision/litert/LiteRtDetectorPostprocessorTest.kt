package com.cardviper.app.vision.litert

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class LiteRtDetectorPostprocessorTest {
    @Test fun reversesLandscapeLetterboxToSourcePixels() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            8,
            4,
            4,
            4,
            ResizePolicy.LETTERBOX,
        )
        val result = LiteRtDetectorPostprocessor.process(
            DetectorOutputs(
                boxes = floatArrayOf(0.25f, 0.25f, 0.75f, 0.75f),
                scores = floatArrayOf(0.9f),
                classes = floatArrayOf(0f),
                count = 1f,
            ),
            detectorEmbeddedContract(),
            transform,
        ).single()
        assertEquals(2f, result.x, 0.0001f)
        assertEquals(0f, result.y, 0.0001f)
        assertEquals(4f, result.width, 0.0001f)
        assertEquals(4f, result.height, 0.0001f)
    }

    @Test fun reversesPortraitLetterboxToSourcePixels() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            4,
            8,
            4,
            4,
            ResizePolicy.LETTERBOX,
        )
        val result = LiteRtDetectorPostprocessor.process(
            DetectorOutputs(
                floatArrayOf(0.25f, 0.25f, 0.75f, 0.75f),
                floatArrayOf(0.8f),
                floatArrayOf(0f),
                1f,
            ),
            detectorEmbeddedContract(),
            transform,
        ).single()
        assertEquals(0f, result.x, 0.0001f)
        assertEquals(2f, result.y, 0.0001f)
        assertEquals(4f, result.width, 0.0001f)
        assertEquals(4f, result.height, 0.0001f)
    }

    @Test fun clampsBoxesThatCrossSourceEdges() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            8,
            4,
            4,
            4,
            ResizePolicy.LETTERBOX,
        )
        val result = LiteRtDetectorPostprocessor.process(
            DetectorOutputs(
                floatArrayOf(-1f, -1f, 2f, 2f),
                floatArrayOf(0.9f),
                floatArrayOf(0f),
                1f,
            ),
            detectorEmbeddedContract(),
            transform,
        ).single()
        assertEquals(0f, result.x, 0f)
        assertEquals(0f, result.y, 0f)
        assertEquals(8f, result.width, 0f)
        assertEquals(4f, result.height, 0f)
    }

    @Test fun rejectsZeroAreaAfterClampingBySkippingCandidate() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            8,
            4,
            4,
            4,
            ResizePolicy.LETTERBOX,
        )
        val result = LiteRtDetectorPostprocessor.process(
            DetectorOutputs(
                floatArrayOf(0.5f, 0.5f, 0.5f, 0.8f),
                floatArrayOf(0.9f),
                floatArrayOf(0f),
                1f,
            ),
            detectorEmbeddedContract(),
            transform,
        )
        assertTrue(result.isEmpty())
    }

    @Test fun appliesScoreThreshold() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            4,
            4,
            4,
            4,
            ResizePolicy.STRETCH,
        )
        val result = LiteRtDetectorPostprocessor.process(
            DetectorOutputs(
                floatArrayOf(0f, 0f, 1f, 1f),
                floatArrayOf(0.49f),
                floatArrayOf(0f),
                1f,
            ),
            detectorEmbeddedContract(),
            transform,
        )
        assertTrue(result.isEmpty())
    }

    @Test fun obeysDetectionCount() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            4,
            4,
            4,
            4,
            ResizePolicy.STRETCH,
        )
        val result = LiteRtDetectorPostprocessor.process(
            DetectorOutputs(
                boxes = floatArrayOf(
                    0f, 0f, 0.5f, 0.5f,
                    0.5f, 0.5f, 1f, 1f,
                ),
                scores = floatArrayOf(0.9f, 0.9f),
                classes = floatArrayOf(0f, 0f),
                count = 1f,
            ),
            detectorEmbeddedContract(),
            transform,
        )
        assertEquals(1, result.size)
    }

    @Test fun rejectsUnexpectedDetectorClass() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            4,
            4,
            4,
            4,
            ResizePolicy.STRETCH,
        )
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtDetectorPostprocessor.process(
                DetectorOutputs(
                    floatArrayOf(0f, 0f, 1f, 1f),
                    floatArrayOf(0.9f),
                    floatArrayOf(1f),
                    1f,
                ),
                detectorEmbeddedContract(),
                transform,
            )
        }
    }

    @Test fun rejectsMalformedTensorLengths() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            4,
            4,
            4,
            4,
            ResizePolicy.STRETCH,
        )
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtDetectorPostprocessor.process(
                DetectorOutputs(
                    floatArrayOf(0f, 0f, 1f),
                    floatArrayOf(0.9f),
                    floatArrayOf(0f),
                    1f,
                ),
                detectorEmbeddedContract(),
                transform,
            )
        }
    }

    @Test fun embeddedNmsPreservesAlreadySelectedDetections() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            4,
            4,
            4,
            4,
            ResizePolicy.STRETCH,
        )
        val result = LiteRtDetectorPostprocessor.process(
            DetectorOutputs(
                boxes = floatArrayOf(
                    0f, 0f, 1f, 1f,
                    0f, 0f, 1f, 1f,
                ),
                scores = floatArrayOf(0.9f, 0.8f),
                classes = floatArrayOf(0f, 0f),
                count = 2f,
            ),
            detectorEmbeddedContract(),
            transform,
        )
        assertEquals(2, result.size)
    }

    @Test fun hostNmsSuppressesOverlappingLowerScoreBox() {
        val transform = LiteRtImagePreprocessor.buildTransform(
            4,
            4,
            4,
            4,
            ResizePolicy.STRETCH,
        )
        val result = LiteRtDetectorPostprocessor.process(
            DetectorOutputs(
                boxes = floatArrayOf(
                    0f, 0f, 1f, 1f,
                    0.05f, 0.05f, 0.95f, 0.95f,
                ),
                scores = floatArrayOf(0.9f, 0.8f),
                classes = floatArrayOf(0f, 0f),
                count = null,
            ),
            detectorHostContract(),
            transform,
        )
        assertEquals(1, result.size)
        assertEquals(0.9f, result.single().confidence, 0f)
    }

    @Test fun supportsExplicitModelPixelXyxyContract() {
        val base = detectorEmbeddedContract()
        val contract = base.copy(
            output = base.output.copy(
                boxOrder = BoxOrder.XYXY,
                boxUnits = BoxUnits.MODEL_PIXELS,
            ),
        )
        val transform = LiteRtImagePreprocessor.buildTransform(
            4,
            4,
            4,
            4,
            ResizePolicy.STRETCH,
        )
        val result = LiteRtDetectorPostprocessor.process(
            DetectorOutputs(
                floatArrayOf(1f, 1f, 3f, 3f),
                floatArrayOf(1f),
                floatArrayOf(0f),
                1f,
            ),
            contract,
            transform,
        ).single()
        assertEquals(1f, result.x, 0f)
        assertEquals(1f, result.y, 0f)
        assertEquals(2f, result.width, 0f)
        assertEquals(2f, result.height, 0f)
    }
}
