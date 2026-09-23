package com.cardviper.app.vision

interface CardRecognizer {
    fun recognize(crop: ByteArray): CardRecognition?
}
