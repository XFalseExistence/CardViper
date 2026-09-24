package com.cardviper.app.vision

/** Honest production boundary until a detector model is installed. */
class UnavailableCardDetector : CardDetector {
    override suspend fun detect(image: VisionImage): List<CardCandidate> =
        throw VisionModelUnavailableException("Card detector model is not installed")
}
