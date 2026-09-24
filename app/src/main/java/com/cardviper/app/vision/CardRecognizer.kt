package com.cardviper.app.vision

interface CardRecognizer {
    suspend fun recognize(crop: VisionImage): CardRecognition
}
