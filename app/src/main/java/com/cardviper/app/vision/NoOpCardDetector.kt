package com.cardviper.app.vision

/** Deterministic empty fake; not a substitute for a missing production model. */
class NoOpCardDetector : CardDetector {
    override suspend fun detect(image: VisionImage): List<CardCandidate> = emptyList()
}
