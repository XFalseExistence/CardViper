package com.cardviper.app.vision.litert

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class LiteRtModelContractTest {
    private val classifier = resourceText("classifier-contract.json")
    private val detector = resourceText("detector-embedded-contract.json")
    private val q = 34.toChar()

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

    @Test fun rejectsUnsupportedSchema() = fails(
        classifier.replace(q + "schema_version" + q + ": 1", q + "schema_version" + q + ": 2"),
    )

    @Test fun rejectsMissingArtifactHash() = fails(
        classifier.replace(q + "sha256" + q, q + "missing_sha256" + q),
    )

    @Test fun rejectsInvalidArtifactHash() = fails(classifier.replace(
        "94ee059335e587e501cc4bf90613e0814f00a7b08bc7c648fd865a2af6a22cc2",
        "not-a-sha",
    ))

    @Test fun rejectsNonpositiveInputDimension() = fails(
        classifier.replace(q + "width" + q + ": 2", q + "width" + q + ": 0"),
    )

    @Test fun rejectsUnsupportedDtype() = fails(classifier.replace("FLOAT32", "INT8"))

    @Test fun rejectsMissingPreprocessing() =
        fails(classifier.replace("preprocessing", "missing_preprocessing"))

    @Test fun rejectsInvalidLayout() = fails(classifier.replace("NHWC", "NCHW"))

    @Test fun rejectsDuplicateClassifierLabels() = fails(
        classifier.replaceFirst(q + "AD" + q, q + "AC" + q),
    )

    @Test fun rejectsClassifierLabelCountOtherThan53() {
        val backLine = "      " + q + "BACK" + q + 10.toChar()
        fails(classifier.replace(backLine, ""))
    }

    @Test fun rejectsBackOutsideCanonicalIndex52() {
        val old = "      " + q + "KS" + q + "," + 10.toChar() +
            "      " + q + "BACK" + q
        val swapped = "      " + q + "BACK" + q + "," + 10.toChar() +
            "      " + q + "KS" + q
        fails(classifier.replace(old, swapped))
    }

    @Test fun rejectsNoncanonicalClassifierLabelOrder() {
        val old = "      " + q + "AC" + q + "," + 10.toChar() +
            "      " + q + "AD" + q
        val swapped = "      " + q + "AD" + q + "," + 10.toChar() +
            "      " + q + "AC" + q
        fails(classifier.replace(old, swapped))
    }

    @Test fun rejectsDetectorSemanticClassesOtherThanCard() =
        fails(detector.replace("CARD", "PLAYING_CARD"))

    @Test fun rejectsMissingDetectorBoxSemantics() =
        fails(detector.replace("box_order", "missing_box_order"))

    @Test fun rejectsMissingNmsOwnership() =
        fails(detector.replace("nms_ownership", "missing_nms_ownership"))

    @Test fun rejectsMissingResizePolicy() =
        fails(detector.replace("policy", "missing_policy"))

    @Test fun rejectsUnknownEnumValue() =
        fails(detector.replace("LETTERBOX", "MAGIC_RESIZE"))

    @Test fun rejectsDuplicateJsonKeys() {
        val key = q + "schema_version" + q + ": 1,"
        fails(classifier.replace(key, key + " " + key))
    }

    private fun fails(json: String) {
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtModelContract.parse(json)
        }
    }
}
