package com.cardviper.app.vision.litert

import com.cardviper.app.vision.CardLabelCodec
import com.cardviper.app.vision.CardRecognition
import com.cardviper.app.vision.RankedIdentity

object LiteRtClassifierPostprocessor {
    fun process(
        scores: FloatArray,
        contract: LiteRtModelContract.Classifier,
        alternativeCount: Int = 3,
    ): CardRecognition {
        require(alternativeCount >= 0) { "alternativeCount cannot be negative" }
        require(contract.labels == LiteRtModelContract.canonicalClassifierLabels) {
            "Classifier label order differs from the canonical export order"
        }
        require(scores.size == contract.labels.size) {
            "Classifier output width ${scores.size} does not match ${contract.labels.size} labels"
        }
        scores.forEachIndexed { index, score ->
            require(score.isFinite()) { "Classifier score[$index] must be finite" }
            require(score in 0f..1f) { "Classifier score[$index] must be in 0..1" }
        }
        val ordered = scores.indices.sortedWith(
            compareByDescending<Int> { scores[it] }.thenBy { it },
        )
        val best = ordered.first()
        val alternatives = ordered.drop(1).take(alternativeCount).map { index ->
            RankedIdentity(CardLabelCodec.decode(contract.labels[index]), scores[index])
        }
        return CardRecognition(
            identity = CardLabelCodec.decode(contract.labels[best]),
            confidence = scores[best],
            alternatives = alternatives,
        )
    }
}
