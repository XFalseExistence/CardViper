package com.cardviper.app.vision

enum class RecognitionStage {
    DETECTING,
    CLASSIFYING,
}

data class RecognitionProgress(
    val stage: RecognitionStage,
    val completed: Int = 0,
    val total: Int = 0,
) {
    init {
        require(completed >= 0 && total >= 0) { "Progress counters cannot be negative" }
    }
}

interface ImageRecognitionEngine {
    suspend fun recognize(
        image: VisionImage,
        onProgress: (RecognitionProgress) -> Unit = {},
    ): ImageRecognitionResult
}
