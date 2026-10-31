package com.cardviper.app.vision.litert

import com.cardviper.app.vision.CardCandidate
import com.cardviper.app.vision.CardDetector
import com.cardviper.app.vision.DetectorInferenceException
import com.cardviper.app.vision.VisionImage
import com.cardviper.app.vision.VisionModelUnavailableException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext

/** Contract-validated detector adapter. All box and NMS semantics come from the supplied contract. */
class LiteRtCardDetector(
    private val contract: LiteRtModelContract,
    modelBytes: ByteArray,
    runtime: LiteRtRuntime,
) : CardDetector, AutoCloseable {
    private val session: LiteRtSession

    init {
        try {
            require(contract.kind == ModelKind.DETECTOR) { "Detector contract is required" }
            session = LiteRtRuntimeBoundary.openValidated(runtime, modelBytes.copyOf(), null, contract)
        } catch (failure: CancellationException) {
            throw failure
        } catch (failure: VisionModelUnavailableException) {
            throw failure
        } catch (failure: Exception) {
            throw VisionModelUnavailableException("Detector model initialization failed: ${failure.message}", failure)
        }
    }

    override suspend fun detect(image: VisionImage): List<CardCandidate> {
        try {
            coroutineContext.ensureActive()
            val input = LiteRtImagePreprocessor.preprocess(image, contract)
            coroutineContext.ensureActive()
            val tensors = session.run(input)
            val detections = LiteRtDetectorPostprocessor.process(contract, tensors, image.width, image.height)
            coroutineContext.ensureActive()
            return detections.map { detected ->
                CardCandidate(
                    x = detected.box.left,
                    y = detected.box.top,
                    width = detected.box.width,
                    height = detected.box.height,
                    confidence = detected.confidence,
                )
            }
        } catch (failure: CancellationException) {
            throw failure
        } catch (failure: VisionModelUnavailableException) {
            throw failure
        } catch (failure: DetectorInferenceException) {
            throw failure
        } catch (failure: Exception) {
            throw DetectorInferenceException("Card detector inference failed", failure)
        }
    }

    override fun close() = session.close()

    companion object {
        fun fromBundle(bundle: LiteRtModelBundle, runtime: LiteRtRuntime): LiteRtCardDetector {
            val contract = parseLiteRtBundleContract(bundle, ModelKind.DETECTOR)
            return LiteRtCardDetector(contract, bundle.modelBytes, runtime)
        }

        fun fromLoader(loader: LiteRtModelBundleLoader, runtime: LiteRtRuntime): LiteRtCardDetector =
            fromBundle(loadLiteRtModelBundle(loader), runtime)
    }
}

internal fun parseLiteRtBundleContract(bundle: LiteRtModelBundle, expectedKind: ModelKind): LiteRtModelContract = try {
    if (expectedKind == ModelKind.DETECTOR) require(bundle.labelsBytes == null) {
        "Detector model bundle cannot include classifier labels"
    }
    LiteRtModelContract.parse(bundle.contractJson).also { contract ->
        require(contract.kind == expectedKind) { "$expectedKind contract is required" }
    }
} catch (failure: CancellationException) {
    throw failure
} catch (failure: VisionModelUnavailableException) {
    throw failure
} catch (failure: Exception) {
    throw VisionModelUnavailableException("$expectedKind model bundle contract is invalid: ${failure.message}", failure)
}
