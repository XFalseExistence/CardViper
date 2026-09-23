package com.cardviper.app.vision

class NoOpCardDetector : CardDetector {
    override fun detect(frame: ByteArray): List<CardCandidate> = emptyList()
}
