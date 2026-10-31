package com.cardviper.app.vision.litert

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class LiteRtModelContractTest {
    private val classifier = resourceText("classifier-contract.json")
    private val detector = resourceText("detector-embedded-contract.json")

    @Test fun parsesStrictClassifierFixture() {
        val parsed = LiteRtModelContract.parse(classifier) as LiteRtModelContract.Classifier
        assertEquals(53, parsed.labels.size)
        assertEquals("BACK", parsed.labels[52])
        assertEquals(TensorDType.FLOAT32, parsed.input.dtype)
    }

    @Test fun parsesStrictDetectorFixture() {
        val parsed = LiteRtModelContract.parse(detector) as LiteRtModelContract.Detector
        assertEquals(listOf("CARD"), parsed.semanticClasses)
        assertEquals(NmsOwnership.EMBEDDED, parsed.output.nmsOwnership)
    }

    @Test fun rejectsUnsupportedSchema() =
        fails(classifier.replace("\"schema_version\": 1", "\"schema_version\": 2"))

    @Test fun rejectsMissingArtifactHash() =
        fails(classifier.replace("\"sha256\"", "\"missing_sha256\""))

    @Test fun rejectsInvalidArtifactHash() = fails(classifier.replace(
        "94ee059335e587e501cc4bf90613e0814f00a7b08bc7c648fd865a2af6a22cc2",
        "not-a-sha",
    ))

    @Test fun rejectsNonpositiveInputDimension() =
        fails(classifier.replace("\"width\": 2", "\"width\": 0"))

    @Test fun rejectsUnsupportedDtype() =
        fails(classifier.replace("\"dtype\": \"FLOAT32\"", "\"dtype\": \"INT8\""))

    @Test fun rejectsMissingPreprocessing() =
        fails(classifier.replace("\"preprocessing\"", "\"missing_preprocessing\""))

    @Test fun rejectsInvalidLayout() =
        fails(classifier.replace("\"layout\": \"NHWC\"", "\"layout\": \"NCHW\""))

    @Test fun rejectsDuplicateClassifierLabels() =
        fails(classifier.replaceFirst("\"AD\"", "\"AC\""))

    @Test fun rejectsClassifierLabelCountOtherThan53() =
        fails(classifier.replace("      \"BACK\"\n", ""))

    @Test fun rejectsBackOutsideCanonicalIndex52() = fails(
        classifier.replace(
            "      \"KS\",\n      \"BACK\"",
            "      \"BACK\",\n      \"KS\"",
        ),
    )

    @Test fun rejectsNoncanonicalClassifierLabelOrder() = fails(
        classifier.replace(
            "      \"AC\",\n      \"AD\"",
            "      \"AD\",\n      \"AC\"",
        ),
    )

    @Test fun rejectsDetectorSemanticClassesOtherThanCard() =
        fails(detector.replace("\"CARD\"", "\"PLAYING_CARD\""))

    @Test fun rejectsMissingDetectorBoxSemantics() =
        fails(detector.replace("\"box_order\"", "\"missing_box_order\""))

    @Test fun rejectsMissingNmsOwnership() =
        fails(detector.replace("\"nms_ownership\"", "\"missing_nms_ownership\""))

    @Test fun rejectsMissingResizePolicy() =
        fails(detector.replace("\"policy\"", "\"missing_policy\""))

    @Test fun rejectsUnknownEnumValue() =
        fails(detector.replace("\"LETTERBOX\"", "\"MAGIC_RESIZE\""))

    @Test fun rejectsDuplicateJsonKeys() = fails(classifier.replace(
        "\"schema_version\": 1,",
        "\"schema_version\": 1, \"schema_version\": 1,",
    ))

    private fun fails(json: String) {
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtModelContract.parse(json)
        }
    }
}
