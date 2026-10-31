package com.cardviper.app.vision.litert

import org.junit.Assert.assertThrows
import org.junit.Test

class LiteRtArtifactValidatorTest {
    @Test fun validatesModelHashSizeAndInputMetadata() {
        val contract = classifierContract()
        LiteRtArtifactValidator.validateModelBytes("TEST".toByteArray(), contract)
        LiteRtArtifactValidator.validateInputMetadata(
            RuntimeInputMetadata(2, 2, 3, TensorDType.FLOAT32, TensorLayout.NHWC),
            contract,
        )
    }

    @Test fun rejectsWrongModelBytes() {
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtArtifactValidator.validateModelBytes("FAIL".toByteArray(), classifierContract())
        }
    }

    @Test fun rejectsRuntimeInputMetadataDrift() {
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtArtifactValidator.validateInputMetadata(
                RuntimeInputMetadata(3, 2, 3, TensorDType.FLOAT32, TensorLayout.NHWC),
                classifierContract(),
            )
        }
    }

    @Test fun validatesLabelsHashAndExactOrder() {
        LiteRtArtifactValidator.validateClassifierLabels(
            resourceText("labels.txt").toByteArray(),
            classifierContract(),
        )
    }

    @Test fun rejectsLabelsContentOrOrderDrift() {
        val changed = resourceText("labels.txt")
            .replaceFirst("AC\nAD", "AD\nAC")
            .toByteArray()
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtArtifactValidator.validateClassifierLabels(changed, classifierContract())
        }
    }
}
