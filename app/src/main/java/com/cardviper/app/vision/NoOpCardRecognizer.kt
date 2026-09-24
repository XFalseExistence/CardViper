package com.cardviper.app.vision

class NoOpCardRecognizer : CardRecognizer {
    override suspend fun recognize(crop: VisionImage): CardRecognition =
        throw VisionModelUnavailableException("Card classifier is not configured")
}
