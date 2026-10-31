package com.cardviper.app.vision.litert

import com.cardviper.app.vision.VisionModelUnavailableException
import kotlinx.coroutines.CancellationException

/** Contract and bytes supplied by a future offline model source. Arrays are copied at both boundaries. */
class LiteRtModelBundle(
    val contractJson: String,
    modelBytes: ByteArray,
    labelsBytes: ByteArray?,
) {
    private val storedModelBytes = modelBytes.copyOf()
    private val storedLabelsBytes = labelsBytes?.copyOf()

    val modelBytes: ByteArray get() = storedModelBytes.copyOf()
    val labelsBytes: ByteArray? get() = storedLabelsBytes?.copyOf()
}

fun interface LiteRtModelBundleLoader {
    /** Missing local bundle content must be reported as VisionModelUnavailableException by the loader. */
    fun load(): LiteRtModelBundle
}

/** JVM-only source for tests and caller-supplied bytes; it performs no filesystem or network access. */
class InMemoryLiteRtModelBundle(bundle: LiteRtModelBundle) : LiteRtModelBundleLoader {
    private val stored = LiteRtModelBundle(bundle.contractJson, bundle.modelBytes, bundle.labelsBytes)
    override fun load(): LiteRtModelBundle = LiteRtModelBundle(stored.contractJson, stored.modelBytes, stored.labelsBytes)
}

internal fun loadLiteRtModelBundle(loader: LiteRtModelBundleLoader): LiteRtModelBundle = try {
    loader.load()
} catch (failure: CancellationException) {
    throw failure
} catch (failure: VisionModelUnavailableException) {
    throw failure
} catch (failure: Exception) {
    throw VisionModelUnavailableException("LiteRT model bundle is unavailable: ${failure.message}", failure)
}
