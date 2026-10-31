package com.cardviper.app.vision.litert

/** A runtime-neutral, dependency-injected seam. No LiteRT implementation is linked here. */
fun interface LiteRtRuntime {
    fun open(modelBytes: ByteArray, contract: LiteRtModelContract): LiteRtSession
}

interface LiteRtSession : AutoCloseable {
    val signature: TensorSignature
    fun run(input: PreprocessedImage): Map<String, FloatArray>
}

/** Compatibility names retained for the initial B4 contract-boundary API. */
typealias LiteRtRuntimeFactory = LiteRtRuntime
typealias LiteRtRuntimeSession = LiteRtSession
