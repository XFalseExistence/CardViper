package com.cardviper.app.vision

import com.cardviper.app.blackjack.CountStrategy
import com.cardviper.app.blackjack.Kiss3Strategy
import com.cardviper.app.blackjack.KoStrategy
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext
import kotlin.math.ceil
import kotlin.math.floor

/** Image-local diagnostic inference; never writes observations to a shoe or ledger. */
class DefaultImageRecognitionEngine(
    private val detector: CardDetector,
    private val recognizer: CardRecognizer,
    private val confidentThreshold: Float = 0.80f,
    private val cropPaddingFraction: Float = 0.10f,
) : ImageRecognitionEngine {
    init {
        require(confidentThreshold.isFinite() && confidentThreshold in 0f..1f) {
            "Confidence threshold must be between zero and one"
        }
        require(cropPaddingFraction.isFinite() && cropPaddingFraction in 0f..1f) {
            "Crop padding fraction must be between zero and one"
        }
    }

    override suspend fun recognize(
        image: VisionImage,
        onProgress: (RecognitionProgress) -> Unit,
    ): ImageRecognitionResult {
        coroutineContext.ensureActive()
        onProgress(RecognitionProgress(RecognitionStage.DETECTING))
        val candidates = try {
            detector.detect(image)
        } catch (failure: CancellationException) {
            throw failure
        } catch (failure: VisionModelUnavailableException) {
            throw failure
        } catch (failure: DetectorInferenceException) {
            throw failure
        } catch (failure: Exception) {
            throw DetectorInferenceException("Card detector inference failed", failure)
        }
        val ordered = candidates.sortedWith(
            compareBy<CardCandidate> { it.y }.thenBy { it.x }.thenByDescending { it.confidence },
        )
        // Keep only geometry here, so full-resolution RGB crops exist one at a time.
        val valid = ordered.mapNotNull { candidate ->
            cropRect(candidate, image)?.let { rect -> candidate to rect }
        }
        val observations = mutableListOf<ImageCardObservation>()
        if (valid.isNotEmpty()) {
            onProgress(RecognitionProgress(RecognitionStage.CLASSIFYING, 0, valid.size))
        }
        for ((candidate, rect) in valid) {
            coroutineContext.ensureActive()
            // Runtime-specific resizing/normalization belongs behind CardRecognizer.
            val crop = image.crop(rect)
            val recognition = try {
                recognizer.recognize(crop)
            } catch (failure: CancellationException) {
                throw failure
            } catch (failure: VisionModelUnavailableException) {
                throw failure
            } catch (failure: ClassifierInferenceException) {
                throw failure
            } catch (failure: Exception) {
                throw ClassifierInferenceException("Card classifier inference failed", failure)
            }
            observations += ImageCardObservation(
                candidate = candidate,
                cropRect = rect,
                identity = recognition.identity,
                classifierConfidence = recognition.confidence,
                uncertain = recognition.confidence < confidentThreshold,
                alternatives = recognition.alternatives.toList(),
            )
            onProgress(RecognitionProgress(RecognitionStage.CLASSIFYING, observations.size, valid.size))
        }
        coroutineContext.ensureActive()
        return ImageRecognitionResult(
            observations = observations.toList(),
            koDelta = previewDelta(KoStrategy(), observations),
            kiss3Delta = previewDelta(Kiss3Strategy(), observations),
            uncertainCount = observations.count { it.uncertain },
            skippedCandidates = candidates.size - valid.size,
        )
    }

    private fun cropRect(candidate: CardCandidate, image: VisionImage): PixelRect? {
        if (candidate.width <= 0f || candidate.height <= 0f) return null
        val padX = (candidate.width * cropPaddingFraction).toDouble()
        val padY = (candidate.height * cropPaddingFraction).toDouble()
        // Double arithmetic prevents overflow when adding large finite detector coordinates.
        val left = floor(candidate.x.toDouble() - padX).coerceIn(0.0, image.width.toDouble()).toInt()
        val top = floor(candidate.y.toDouble() - padY).coerceIn(0.0, image.height.toDouble()).toInt()
        val right = ceil(candidate.x.toDouble() + candidate.width + padX)
            .coerceIn(0.0, image.width.toDouble()).toInt()
        val bottom = ceil(candidate.y.toDouble() + candidate.height + padY)
            .coerceIn(0.0, image.height.toDouble()).toInt()
        return if (right > left && bottom > top) PixelRect(left, top, right, bottom) else null
    }

    private fun previewDelta(strategy: CountStrategy, observations: List<ImageCardObservation>): Int =
        observations.sumOf { observation ->
            when (val identity = observation.identity) {
                CardIdentity.Back -> 0
                is CardIdentity.Face -> strategy.valueOf(identity.card) ?: 0
            }
        }
}
