package com.cardviper.app.vision.litert

import com.cardviper.app.vision.ImageRecognitionEngine
import com.cardviper.app.vision.CardIdentity
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Test

class SessionADemoVisionFactoryTest {
    @Test fun standardBuildSelectionNeverInitializesDemoAssets() {
        val standard = ImageRecognitionEngine { _, _ -> error("unused") }
        val selected = SessionADemoVisionFactory.select(false, { standard }) {
            error("demo assets must not load in a standard build")
        }

        assertSame(standard, selected)
    }

    @Test fun demoFaceContractAcceptsExactlyTheCanonical52FaceLabels() {
        val labels = canonicalAdapterLabels().dropLast(1)
        val model = "demo-model".toByteArray()
        val labelBytes = (labels.joinToString("\n") + "\n").toByteArray()
        val contract = LiteRtModelContract.parse("""
            {"schema_version":1,"model_kind":"SESSION_A_FACE_CLASSIFIER",
             "artifact":{"sha256":"${sha256(model)}"},
             "input":{"name":"rgb_0_255","shape":[1,224,224,3],"layout":"NHWC","dtype":"FLOAT32"},
             "preprocessing":{"resize_policy":"STRETCH","resize_filter":"BILINEAR","channel_order":"RGB","scale":1.0,"offset":0.0,"pad_value":0},
             "outputs":[{"name":"scores","role":"CLASSIFIER_SCORES","shape":[1,52],"dtype":"FLOAT32"}],
             "labels":${labels.joinToString(prefix = "[", postfix = "]") { "\"$it\"" }},
             "labels_sha256":"${sha256(labelBytes)}","classifier":{"top_k":3,"score_semantics":"PROBABILITIES"},
             "detector":null,"session_a_demo":{"usage":"SESSION_A_DEMO_ONLY","independence":"NOT_INDEPENDENT","face_up_only":true,"back_supported":false}}
        """.trimIndent())

        assertSame(ModelKind.SESSION_A_FACE_CLASSIFIER, contract.kind)
        assertEquals(52, contract.labels.size)
        assertThrows(IllegalArgumentException::class.java) {
            LiteRtModelContract.parse("""{"schema_version":1,"model_kind":"SESSION_A_FACE_CLASSIFIER"}""")
        }
    }

    @Test fun lowConfidenceSessionAFaceScoresBecomeUnknownInsteadOfAGuessedFace() {
        val labels = canonicalAdapterLabels().dropLast(1)
        val model = "demo-model".toByteArray()
        val labelBytes = (labels.joinToString("\n") + "\n").toByteArray()
        val contract = demoFaceContract(model, labelBytes, labels)
        val scores = FloatArray(52) { 0.01f }.also { it[3] = 0.55f }

        val prediction = SessionADemoFacePostprocessor.process(scores, contract)

        assertEquals(CardIdentity.Unknown, prediction.identity)
        assertEquals(0.55f, prediction.confidence, 0f)
    }

    private fun demoFaceContract(model: ByteArray, labelsBytes: ByteArray, labels: List<String>): LiteRtModelContract =
        LiteRtModelContract.parse("""
            {"schema_version":1,"model_kind":"SESSION_A_FACE_CLASSIFIER",
             "artifact":{"sha256":"${sha256(model)}"},
             "input":{"name":"rgb_0_255","shape":[1,224,224,3],"layout":"NHWC","dtype":"FLOAT32"},
             "preprocessing":{"resize_policy":"STRETCH","resize_filter":"BILINEAR","channel_order":"RGB","scale":1.0,"offset":0.0,"pad_value":0},
             "outputs":[{"name":"scores","role":"CLASSIFIER_SCORES","shape":[1,52],"dtype":"FLOAT32"}],
             "labels":${labels.joinToString(prefix = "[", postfix = "]") { "\"$it\"" }},
             "labels_sha256":"${sha256(labelsBytes)}","classifier":{"top_k":3,"score_semantics":"PROBABILITIES"},
             "detector":null,"session_a_demo":{"usage":"SESSION_A_DEMO_ONLY","independence":"NOT_INDEPENDENT","face_up_only":true,"back_supported":false}}
        """.trimIndent())

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }
}
