package com.cardviper.app.vision.litert

import java.nio.charset.StandardCharsets
import java.security.MessageDigest

data class ObservedTensor(val name: String, val shape: List<Int>, val dtype: TensorDataType)
data class TensorSignature(val input: ObservedTensor, val outputs: List<ObservedTensor>)

/** Verifies local bytes and observed runtime metadata before any future adapter can open a model. */
object LiteRtArtifactValidator {
    fun validate(
        contract: LiteRtModelContract,
        modelBytes: ByteArray,
        labelsBytes: ByteArray?,
        observed: TensorSignature,
    ) {
        validateDeclaredBytes(contract, modelBytes, labelsBytes)
        validateObservedSignature(contract, observed)
    }

    fun validateDeclaredBytes(contract: LiteRtModelContract, modelBytes: ByteArray, labelsBytes: ByteArray?) {
        require(modelBytes.isNotEmpty()) { "Model artifact is empty" }
        require(sha256(modelBytes).equals(contract.artifactSha256, ignoreCase = true)) { "Model artifact SHA-256 mismatch" }
        if (contract.kind == ModelKind.CLASSIFIER) {
            require(labelsBytes != null) { "Classifier labels file is required" }
            val labelHash = contract.labelsSha256 ?: throw IllegalArgumentException("Classifier labels SHA-256 is required")
            require(sha256(labelsBytes).equals(labelHash, ignoreCase = true)) { "Classifier labels SHA-256 mismatch" }
            val text = String(labelsBytes, StandardCharsets.UTF_8)
            require(!text.startsWith('\uFEFF') && !text.contains('\r') && text.endsWith('\n')) { "Classifier labels must be UTF-8 LF lines with a final newline" }
            val labels = text.dropLast(1).split('\n')
            require(labels == contract.labels) { "Classifier labels file does not match contract order" }
        } else {
            require(labelsBytes == null) { "Detector does not consume a classifier labels file" }
        }
    }

    fun validateObservedSignature(contract: LiteRtModelContract, observed: TensorSignature) {
        val expectedInput = ObservedTensor(contract.input.name, contract.input.shape, contract.input.dtype)
        val expectedOutputs = contract.outputs.map { ObservedTensor(it.name, it.shape, it.dtype) }
        require(observed.input == expectedInput) { "Observed model input does not match contract" }
        require(observed.outputs == expectedOutputs) { "Observed model outputs do not match contract" }
    }

    fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { byte -> "%02x".format(byte) }
}
