package com.cardviper.app.vision.litert

import com.cardviper.app.vision.VisionModelUnavailableException
import java.security.MessageDigest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.Assert.assertThrows

class AndroidAssetLiteRtModelBundleTest {
    @Test fun missingModelAssetThrowsUnavailable() {
        val failure = assertThrows(VisionModelUnavailableException::class.java) {
            AndroidAssetLiteRtModelBundle(MapAssetBytes(emptyMap()), "vision", ModelKind.DETECTOR).load()
        }
        assertTrue(failure.message!!.contains("Session A demo asset"))
    }

    @Test fun detectorBytesMustMatchTheSignedManifest() {
        val model = byteArrayOf(1, 2, 3)
        val files = detectorAssets(model).toMutableMap().also { files ->
            files["vision/manifest.json"] = files.getValue("vision/manifest.json")
                .replace(sha256(model), "0".repeat(64)).toByteArray()
        }

        assertThrows(VisionModelUnavailableException::class.java) {
            AndroidAssetLiteRtModelBundle(MapAssetBytes(files), "vision", ModelKind.DETECTOR).load()
        }
    }

    @Test fun verifiedDetectorBundleReturnsOnlyTheExpectedBytes() {
        val model = byteArrayOf(3, 2, 1)
        val bundle = AndroidAssetLiteRtModelBundle(MapAssetBytes(detectorAssets(model)), "vision", ModelKind.DETECTOR).load()

        assertArrayEquals(model, bundle.modelBytes)
        assertTrue(bundle.labelsBytes == null)
    }

    private fun detectorAssets(model: ByteArray): Map<String, ByteArray> {
        val hash = sha256(model)
        val contract = """
            {"schema_version":1,"model_kind":"DETECTOR","artifact":{"sha256":"$hash"},
             "input":{"name":"image","shape":[1,320,320,3],"layout":"NHWC","dtype":"FLOAT32"},
             "preprocessing":{"resize_policy":"STRETCH","resize_filter":"BILINEAR","channel_order":"RGB","scale":1.0,"offset":0.0,"pad_value":0},
             "outputs":[
               {"name":"boxes","role":"DETECTION_BOXES","shape":[1,1,4],"dtype":"FLOAT32"},
               {"name":"scores","role":"DETECTION_SCORES","shape":[1,1],"dtype":"FLOAT32"},
               {"name":"classes","role":"DETECTION_CLASSES","shape":[1,1],"dtype":"FLOAT32"},
               {"name":"count","role":"DETECTION_COUNT","shape":[1],"dtype":"FLOAT32"}],
             "labels":null,"labels_sha256":null,"classifier":null,
             "detector":{"semantic_classes":["CARD"],"card_class_index":0,
               "boxes":{"tensor":"boxes","coordinate_order":"YXYX","coordinate_units":"NORMALIZED"},
               "scores":{"tensor":"scores"},"classes":{"tensor":"classes"},"count":{"tensor":"count"},
               "nms":{"ownership":"EMBEDDED","score_threshold":0.5,"iou_threshold":0.5}}}
        """.trimIndent()
        val manifest = """
            {"schema_version":1,"usage":"SESSION_A_DEMO_ONLY","independence":"NOT_INDEPENDENT",
             "face_up_only":true,"back_supported":false,
             "assets":{"detector":{"file":"detector.tflite","sha256":"$hash","contract":"detector.json"}}}
        """.trimIndent()
        return mapOf(
            "vision/manifest.json" to manifest.toByteArray(),
            "vision/detector.tflite" to model,
            "vision/detector.json" to contract.toByteArray(),
        )
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }
}

private class MapAssetBytes(private val files: Map<String, ByteArray>) : AssetBytes {
    override fun read(path: String): ByteArray = files[path]?.copyOf()
        ?: throw java.io.FileNotFoundException(path)
}
