package com.cardviper.app.ui.imagetest

import com.cardviper.app.vision.ImageRecognitionResult
import com.cardviper.app.vision.RecognitionProgress
import com.cardviper.app.vision.VisionImage

enum class ImageTestPhase { IDLE, LOADING_IMAGE, DETECTING, CLASSIFYING, COMPLETE, NO_CARDS, ERROR }

data class ImageRecognitionUiState(
    val phase: ImageTestPhase = ImageTestPhase.IDLE,
    val source: String? = null,
    val image: VisionImage? = null,
    val result: ImageRecognitionResult? = null,
    val progress: RecognitionProgress? = null,
    val statusText: String = "CHOOSE IMAGE",
)
