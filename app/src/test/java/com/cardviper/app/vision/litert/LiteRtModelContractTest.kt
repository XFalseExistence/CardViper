package com.cardviper.app.vision.litert

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class LiteRtModelContractTest {
    @Test fun parsesTestOnlyClassifierAndDetectorContracts() {
        val classifier = LiteRtModelContract.parse(fixture("classifier"))
        val detector = LiteRtModelContract.parse(fixture("detector"))
        assertEquals(ModelKind.CLASSIFIER, classifier.kind)
        assertEquals(53, classifier.labels.size)
        assertEquals("BACK", classifier.labels[52])
        assertEquals(ModelKind.DETECTOR, detector.kind)
        assertEquals(listOf("CARD"), detector.detector!!.semanticClasses)
    }

    @Test fun rejectsUnsupportedSchemaVersionAndMissingOrMalformedArtifactHash() {
        reject(classifier().replace("\"schema_version\": 1", "\"schema_version\": 2"))
        reject(classifier().replace("\"artifact\": { \"sha256\": \"${ZERO_HASH}\" }", "\"artifact\": {}"))
        reject(classifier().replace("\"sha256\": \"${ZERO_HASH}\"", "\"sha256\": \"\""))
        reject(classifier().replace(ZERO_HASH, "not-a-sha"))
    }

    @Test fun rejectsInvalidDimensionsDtypesAndLayouts() {
        reject(classifier().replace("[1, 2, 2, 3]", "[1, 0, 2, 3]"))
        reject(classifier().replace("[1, 53]", "[1, -1]"))
        reject(classifier().replace("\"dtype\": \"FLOAT32\"", "\"dtype\": \"INT8\""))
        reject(classifier().replace("\"layout\": \"NHWC\"", "\"layout\": \"CHW\""))
    }

    @Test fun rejectsMissingPreprocessingAndUnknownEnumValues() {
        reject(classifier().replace(Regex("\\s*\"preprocessing\": \\{[^}]+\\},"), ""))
        reject(classifier().replace("\"resize_policy\": \"STRETCH\"", "\"resize_policy\": \"MAGIC\""))
    }

    @Test fun rejectsDuplicateOrNoncanonicalClassifierLabelsAndOutputWidths() {
        reject(classifier().replace("\"AD\"", "\"AC\""))
        reject(classifier().replace("\"BACK\"],", "\"AC\"],"))
        reject(classifier().replace("[1, 53]", "[1, 52]"))
    }

    @Test fun rejectsDetectorSemanticClassBoxNmsAndResizeContractGaps() {
        reject(detector().replace("\"semantic_classes\": [\"CARD\"]", "\"semantic_classes\": [\"ACE_SPADES\"]"))
        reject(detector().replace(Regex("\"boxes\": \\{[^}]+\\},"), ""))
        reject(detector().replace("\"ownership\": \"HOST\"", "\"ownership\": \"MAYBE\""))
        reject(detector().replace("\"ownership\": \"HOST\", ", ""))
        reject(detector().replace("\"resize_policy\": \"LETTERBOX\"", "\"resize_policy\": \"\""))
        reject(detector().replace("\"resize_policy\": \"LETTERBOX\", ", ""))
        reject(detector().replace("\"coordinate_order\": \"YXYX\"", "\"coordinate_order\": \"YXZX\""))
    }

    @Test fun rejectsDuplicateOutputNamesAndUnsupportedOutputRoles() {
        reject(detector().replace("\"name\": \"scores\"", "\"name\": \"boxes\""))
        reject(detector().replace("\"role\": \"DETECTION_SCORES\"", "\"role\": \"OTHER\""))
    }

    @Test fun rejectsMissingOrUnknownClassifierScoreSemantics() {
        reject(classifier().replace(", \"score_semantics\": \"PROBABILITIES\"", ""))
        reject(classifier().replace("\"PROBABILITIES\"", "\"LOGITS\""))
    }

    private fun reject(raw: String) {
        assertThrows(IllegalArgumentException::class.java) { LiteRtModelContract.parse(raw) }
    }

    private fun classifier() = fixture("classifier")
    private fun detector() = fixture("detector")
    private fun fixture(name: String) = javaClass.getResource("/vision/$name.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")!!.readText()

    private companion object {
        const val ZERO_HASH = "0000000000000000000000000000000000000000000000000000000000000000"
    }
}
