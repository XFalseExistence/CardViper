package com.cardviper.app.vision.litert

/** Dependency-injected seam for a future standalone LiteRT interpreter; this repository has no runtime implementation yet. */
fun interface LiteRtRuntimeFactory {
    fun open(modelBytes: ByteArray, contract: LiteRtModelContract): LiteRtRuntimeSession
}

interface LiteRtRuntimeSession : AutoCloseable {
    val signature: TensorSignature
    fun run(input: PreprocessedImage): Map<String, FloatArray>
}

/** Hashes bytes before opening; checks observed tensors before returning a usable session. */
object LiteRtRuntimeBoundary {
    fun openValidated(
        factory: LiteRtRuntimeFactory,
        modelBytes: ByteArray,
        labelsBytes: ByteArray?,
        contract: LiteRtModelContract,
    ): LiteRtRuntimeSession {
        LiteRtArtifactValidator.validateDeclaredBytes(contract, modelBytes, labelsBytes)
        val session = factory.open(modelBytes, contract)
        try {
            LiteRtArtifactValidator.validateObservedSignature(contract, session.signature)
            return session
        } catch (failure: Throwable) {
            runCatching { session.close() }
            throw failure
        }
    }
}
