package com.cardviper.app.vision

class NoOpCardRecognizer : CardRecognizer {
    override fun recognize(crop: ByteArray): CardRecognition? = null
}
