package com.cardviper.app.vision.litert

import android.content.Context
import com.cardviper.app.vision.VisionModelUnavailableException
import java.security.MessageDigest
import org.json.JSONObject

/** Minimal byte source so fail-closed asset behavior remains JVM-testable. */
fun interface AssetBytes {
    fun read(path: String): ByteArray
}

private class AndroidAssetBytes(private val context: Context) : AssetBytes {
    override fun read(path: String): ByteArray = context.assets.open(path).use { it.readBytes() }
}

/** Loads one verified Session A model bundle from variant-only APK assets. */
class AndroidAssetLiteRtModelBundle(
    private val assets: AssetBytes,
    assetPrefix: String,
    private val kind: ModelKind,
) : LiteRtModelBundleLoader {
    private val prefix = assetPrefix.trim('/')

    constructor(context: Context, assetPrefix: String, kind: ModelKind) :
        this(AndroidAssetBytes(context), assetPrefix, kind)

    init {
        require(kind == ModelKind.DETECTOR || kind == ModelKind.SESSION_A_FACE_CLASSIFIER) {
            "Session A assets support only the CARD detector and 52-face classifier"
        }
        require(prefix.isNotBlank()) { "Session A asset prefix is required" }
    }

    override fun load(): LiteRtModelBundle = try {
        val manifest = JSONObject(String(read("manifest.json"), Charsets.UTF_8))
        require(manifest.getInt("schema_version") == 1) { "Unsupported Session A manifest version" }
        require(manifest.getString("usage") == "SESSION_A_DEMO_ONLY") { "Session A manifest is not demo-only" }
        require(manifest.getString("independence") == "NOT_INDEPENDENT") { "Session A manifest lacks non-independent disclosure" }
        require(manifest.getBoolean("face_up_only") && !manifest.getBoolean("back_supported")) {
            "Session A manifest must remain face-up-only"
        }
        val key = if (kind == ModelKind.DETECTOR) "detector" else "classifier"
        val entry = manifest.getJSONObject("assets").getJSONObject(key)
        val model = read(entry.getString("file"))
        require(sha256(model).equals(entry.getString("sha256"), ignoreCase = true)) { "Manifest model SHA-256 mismatch" }
        val contractJson = String(read(entry.getString("contract")), Charsets.UTF_8)
        val contract = LiteRtModelContract.parse(contractJson)
        require(contract.kind == kind) { "Manifest contract model kind differs from requested asset" }
        require(sha256(model).equals(contract.artifactSha256, ignoreCase = true)) { "Contract model SHA-256 mismatch" }
        val labels = if (kind == ModelKind.SESSION_A_FACE_CLASSIFIER) {
            val bytes = read(entry.getString("labels"))
            require(sha256(bytes).equals(contract.labelsSha256, ignoreCase = true)) { "Classifier labels SHA-256 mismatch" }
            bytes
        } else {
            require(!entry.has("labels")) { "Detector bundle must not declare classifier labels" }
            null
        }
        LiteRtModelBundle(contractJson, model, labels)
    } catch (failure: VisionModelUnavailableException) {
        throw failure
    } catch (failure: Exception) {
        throw VisionModelUnavailableException("Session A demo asset is unavailable or invalid: ${failure.message}", failure)
    }

    private fun read(name: String): ByteArray {
        require(name.isNotBlank() && !name.contains('/') && !name.contains('\\')) { "Invalid Session A asset filename" }
        return assets.read("$prefix/$name")
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }
}
