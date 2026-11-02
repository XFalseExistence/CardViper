package com.cardviper.app.vision.litert

import com.cardviper.app.vision.CardIdentity
import com.cardviper.app.vision.CardLabelCodec
import com.cardviper.app.vision.CardRecognition
import com.cardviper.app.vision.CardRecognizer
import com.cardviper.app.vision.ClassifierInferenceException
import com.cardviper.app.vision.RankedIdentity
import com.cardviper.app.vision.VisionImage
import com.cardviper.app.vision.VisionModelUnavailableException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext

/** Separate 52-face adapter: it cannot emit a BACK identity. */
class SessionADemoFaceRecognizer(
    private val contract: LiteRtModelContract,
    modelBytes: ByteArray,
    labelsBytes: ByteArray,
    runtime: LiteRtRuntime,
) : CardRecognizer, AutoCloseable {
    private val session: LiteRtSession

    init {
        try {
            require(contract.kind == ModelKind.SESSION_A_FACE_CLASSIFIER) { "Session A face classifier contract is required" }
            session = LiteRtRuntimeBoundary.openValidated(runtime, modelBytes.copyOf(), labelsBytes.copyOf(), contract)
        } catch (failure: CancellationException) {
            throw failure
        } catch (failure: VisionModelUnavailableException) {
            throw failure
        } catch (failure: Exception) {
            throw VisionModelUnavailableException("Session A classifier initialization failed: ${failure.message}", failure)
        }
    }

    override suspend fun recognize(crop: VisionImage): CardRecognition = try {
        coroutineContext.ensureActive()
        val tensors = session.run(LiteRtImagePreprocessor.preprocess(crop, contract))
        require(tensors.keys == contract.outputs.map { it.name }.toSet()) { "Session A classifier output tensors do not match contract" }
        val scores = requireNotNull(tensors[contract.outputs.single().name]) { "Session A classifier scores are missing" }
        val prediction = SessionADemoFacePostprocessor.process(scores, contract)
        coroutineContext.ensureActive()
        CardRecognition(prediction.identity, prediction.confidence, prediction.alternatives)
    } catch (failure: CancellationException) {
        throw failure
    } catch (failure: VisionModelUnavailableException) {
        throw failure
    } catch (failure: ClassifierInferenceException) {
        throw failure
    } catch (failure: Exception) {
        throw ClassifierInferenceException("Session A classifier inference failed", failure)
    }

    override fun close() = session.close()

    companion object {
        fun fromLoader(loader: LiteRtModelBundleLoader, runtime: LiteRtRuntime): SessionADemoFaceRecognizer {
            val bundle = loadLiteRtModelBundle(loader)
            val contract = parseLiteRtBundleContract(bundle, ModelKind.SESSION_A_FACE_CLASSIFIER)
            return SessionADemoFaceRecognizer(contract, bundle.modelBytes, requireNotNull(bundle.labelsBytes), runtime)
        }
    }
}

private data class SessionADemoFacePrediction(
    val identity: CardIdentity,
    val confidence: Float,
    val alternatives: List<RankedIdentity>,
)

internal object SessionADemoFacePostprocessor {
    fun process(scores: FloatArray, contract: LiteRtModelContract): SessionADemoFacePrediction {
        require(contract.kind == ModelKind.SESSION_A_FACE_CLASSIFIER) { "Session A face contract required" }
        require(contract.labels == LiteRtModelContract.sessionADemoFaceLabels && scores.size == 52) { "Session A scores must map to exactly 52 face labels" }
        scores.forEachIndexed { index, score -> require(score.isFinite() && score in 0f..1f) { "Session A score at $index is invalid" } }
        val best = scores.indices.maxWithOrNull(compareBy<Int> { scores[it] }.thenByDescending { it })!!
        fun identity(index: Int): CardIdentity.Face = CardLabelCodec.decode(contract.labels[index]) as CardIdentity.Face
        val confidence = scores[best]
        return SessionADemoFacePrediction(
            if (confidence < MINIMUM_FACE_CONFIDENCE) CardIdentity.Unknown else identity(best), confidence,
            scores.indices.asSequence().filter { it != best }
                .sortedWith(compareByDescending<Int> { scores[it] }.thenBy { it })
                .take(contract.classifier!!.topK - 1)
                .map { index -> RankedIdentity(identity(index), scores[index]) }.toList(),
        )
    }

    private const val MINIMUM_FACE_CONFIDENCE = 0.80f
}
