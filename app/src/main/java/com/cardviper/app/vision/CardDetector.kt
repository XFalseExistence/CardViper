package com.cardviper.app.vision

interface CardDetector {
    suspend fun detect(image: VisionImage): List<CardCandidate>
}
