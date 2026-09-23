package com.cardviper.app.vision

interface CardDetector {
    fun detect(frame: ByteArray): List<CardCandidate>
}
