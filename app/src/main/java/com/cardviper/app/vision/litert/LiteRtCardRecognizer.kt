package com.cardviper.app.vision.litert

import com.cardviper.app.vision.CardRecognition
import com.cardviper.app.vision.CardRecognizer
import com.cardviper.app.vision.ClassifierInferenceException
import com.cardviper.app.vision.RankedIdentity
import com.cardviper.app.vision.VisionImage
import com.cardviper.app.vision.VisionModelUnavailableException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext

/** Contract-validated classifier adapter. It remains inert until constructed with an injected runtime and bundle. */
class LiteRtCardRecognizer(
    private val contract: LiteRtModelContract,
    modelBytes: ByteArray,
    labelsBytes: ByteArray?,
    runtime: LiteRtRuntime,
) : CardRecognizer, AutoCloseable {
    private val session: LiteRtSession

    init {
        try {
            require(contract.kind == ModelKind.CLASSIFIER) { "Classifier contract is required" }
            session = LiteRtRuntimeBoundary.openValidated(runtime, modelBytes.copyOf(), labelsBytes?.copyOf(), contract)
        } catch (failure: CancellationException) {
            throw failure
        } catch (failure: VisionModelUnavailableException) {
            throw failure
        } catch (failure: Exception) {
            throw VisionModelUnavailableException("Classifier model initialization failed: ${failure.message}", failure)
        }
    }

    override suspend fun recognize(crop: VisionImage): CardRecognition {
        try {
            coroutineContext.ensureActive()
            val input = LiteRtImagePreprocessor.preprocess(crop, contract)
            coroutineContext.ensureActive()
            val tensors = session.run(input)
            val expectedNames = contract.outputs.map { it.name }.toSet()
            require(tensors.keys == expectedNames) { "Classifier output tensor names do not match contract" }
            val scoreName = contract.outputs.single().name
            val scores = requireNotNull(tensors[scoreName]) { "Classifier score tensor '$scoreName' is missing" }
            val prediction = LiteRtClassifierPostprocessor.process(scores, contract, contract.labels)
            coroutineContext.ensureActive()
            return CardRecognition(
                identity = prediction.identity,
                confidence = prediction.confidence,
                alternatives = prediction.alternatives.map { RankedIdentity(it.identity, it.confidence) },
            )
        } catch (failure: CancellationException) {
            throw failure
        } catch (failure: VisionModelUnavailableException) {
            throw failure
        } catch (failure: ClassifierInferenceException) {
            throw failure
        } catch (failure: Exception) {
            throw ClassifierInferenceException("Card classifier inference failed", failure)
        }
    }

    override fun close() = session.close()

    companion object {
        fun fromBundle(bundle: LiteRtModelBundle, runtime: LiteRtRuntime): LiteRtCardRecognizer {
            val contract = parseLiteRtBundleContract(bundle, ModelKind.CLASSIFIER)
            return LiteRtCardRecognizer(contract, bundle.modelBytes, bundle.labelsBytes, runtime)
        }

        fun fromLoader(loader: LiteRtModelBundleLoader, runtime: LiteRtRuntime): LiteRtCardRecognizer =
            fromBundle(loadLiteRtModelBundle(loader), runtime)
    }
}
