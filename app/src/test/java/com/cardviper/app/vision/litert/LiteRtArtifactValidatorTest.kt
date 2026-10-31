package com.cardviper.app.vision.litert

import org.junit.Assert.assertThrows
import org.junit.Test
import java.security.MessageDigest

class LiteRtArtifactValidatorTest {
    @Test fun verifiesModelHashLabelsHashAndContractTensorAgreement() {
        val model = "synthetic test model".toByteArray()
        val labels = (canonicalLabels().joinToString("\n") + "\n").toByteArray()
        val parsed = LiteRtModelContract.parse(javaClass.getResource("/vision/classifier.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")!!.readText())
        val contract = parsed.copy(artifactSha256 = sha(model), labelsSha256 = sha(labels))
        LiteRtArtifactValidator.validate(contract, model, labels, signature(contract))
    }

    @Test fun rejectsWrongModelHashLabelsAndTensorMetadata() {
        val model = "synthetic test model".toByteArray()
        val labels = (canonicalLabels().joinToString("\n") + "\n").toByteArray()
        val parsed = LiteRtModelContract.parse(javaClass.getResource("/vision/classifier.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")!!.readText())
        val contract = parsed.copy(artifactSha256 = sha(model), labelsSha256 = sha(labels))
        assertThrows(IllegalArgumentException::class.java) { LiteRtArtifactValidator.validate(contract, "other".toByteArray(), labels, signature(contract)) }
        assertThrows(IllegalArgumentException::class.java) { LiteRtArtifactValidator.validate(contract, model, "wrong\n".toByteArray(), signature(contract)) }
        val bad = signature(contract).copy(input = signature(contract).input.copy(shape = listOf(1, 3, 2, 3)))
        assertThrows(IllegalArgumentException::class.java) { LiteRtArtifactValidator.validate(contract, model, labels, bad) }
    }

    private fun canonicalLabels() = listOf("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
        .flatMap { rank -> listOf("C", "D", "H", "S").map { "$rank$it" } } + "BACK"
    private fun sha(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    private fun signature(contract: LiteRtModelContract) = TensorSignature(
        ObservedTensor(contract.input.name, contract.input.shape, contract.input.dtype),
        contract.outputs.map { ObservedTensor(it.name, it.shape, it.dtype) },
    )
}
