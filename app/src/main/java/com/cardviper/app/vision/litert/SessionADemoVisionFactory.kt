package com.cardviper.app.vision.litert

import android.content.Context
import com.cardviper.app.vision.DefaultImageRecognitionEngine
import com.cardviper.app.vision.ImageRecognitionEngine

/** Creates real local inference only in the explicitly selected Session A variant. */
object SessionADemoVisionFactory {
    fun create(context: Context): ImageRecognitionEngine {
        val runtime = StandaloneLiteRtRuntime()
        val detector = LiteRtCardDetector.fromLoader(
            AndroidAssetLiteRtModelBundle(context, "vision", ModelKind.DETECTOR), runtime,
        )
        val recognizer = SessionADemoFaceRecognizer.fromLoader(
            AndroidAssetLiteRtModelBundle(context, "vision", ModelKind.SESSION_A_FACE_CLASSIFIER), runtime,
        )
        return DefaultImageRecognitionEngine(detector, recognizer)
    }

    internal fun select(
        isSessionADemo: Boolean,
        standard: () -> ImageRecognitionEngine,
        demo: () -> ImageRecognitionEngine,
    ): ImageRecognitionEngine = if (isSessionADemo) demo() else standard()
}
