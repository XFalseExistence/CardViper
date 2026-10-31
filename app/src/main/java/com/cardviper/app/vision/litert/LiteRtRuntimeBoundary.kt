package com.cardviper.app.vision.litert

/** Hashes bytes before opening; checks observed tensors before returning a usable session. */
object LiteRtRuntimeBoundary {
    fun openValidated(
        factory: LiteRtRuntime,
        modelBytes: ByteArray,
        labelsBytes: ByteArray?,
        contract: LiteRtModelContract,
    ): LiteRtRuntimeSession {
        LiteRtArtifactValidator.validateDeclaredBytes(contract, modelBytes, labelsBytes)
        val session = factory.open(modelBytes.copyOf(), contract)
        try {
            LiteRtArtifactValidator.validateObservedSignature(contract, session.signature)
            return session
        } catch (failure: Throwable) {
            runCatching { session.close() }
            throw failure
        }
    }
}
