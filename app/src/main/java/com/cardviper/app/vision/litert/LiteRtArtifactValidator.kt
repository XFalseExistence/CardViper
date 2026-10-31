package com.cardviper.app.vision.litert

import java.security.MessageDigest

data class RuntimeInputMetadata(
    val width: Int,
    val height: Int,
    val channels: Int,
    val dtype: TensorDType,
    val layout: TensorLayout,
)

object LiteRtArtifactValidator {
    fun validateModelBytes(modelBytes: ByteArray, contract: LiteRtModelContract) {
        require(modelBytes.isNotEmpty()) { "Model artifact cannot be empty" }
        require(modelBytes.size.toLong() == contract.artifact.sizeBytes) {
            "Model artifact size does not match contract"
        }
        require(sha256(modelBytes) == contract.artifact.sha256) {
            "Model artifact SHA-256 does not match contract"
        }
    }

    fun validateInputMetadata(metadata: RuntimeInputMetadata, contract: LiteRtModelContract) {
        require(metadata.width == contract.input.width && metadata.height == contract.input.height &&
            metadata.channels == contract.input.channels && metadata.dtype == contract.input.dtype &&
            metadata.layout == contract.input.layout) {
            "Runtime input tensor metadata does not match contract"
        }
    }

    fun validateClassifierLabels(labelBytes: ByteArray, contract: LiteRtModelContract.Classifier) {
        require(sha256(labelBytes) == contract.labelsSha256) { "labels SHA-256 does not match contract" }
        val text = labelBytes.toString(Charsets.UTF_8)
        require(!text.contains('\r')) { "labels.txt must use LF line endings" }
        val labels = text.split('\n').let { if (it.lastOrNull().isNullOrEmpty()) it.dropLast(1) else it }
        require(labels == contract.labels) { "labels.txt order/content does not match classifier contract" }
    }

    fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }
}
