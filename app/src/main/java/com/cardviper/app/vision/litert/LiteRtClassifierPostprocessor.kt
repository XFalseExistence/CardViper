package com.cardviper.app.vision.litert

import com.cardviper.app.vision.CardIdentity
import com.cardviper.app.vision.CardLabelCodec

data class ClassifierAlternative(val label: String, val identity: CardIdentity, val confidence: Float)
data class ClassifierPrediction(
    val label: String,
    val identity: CardIdentity,
    val confidence: Float,
    val alternatives: List<ClassifierAlternative>,
)

/** Maps classifier probabilities only when fixture/real labels exactly match the caller's canonical sequence. */
object LiteRtClassifierPostprocessor {
    fun process(scores: FloatArray, contract: LiteRtModelContract, canonicalLabels: List<String>): ClassifierPrediction {
        require(contract.kind == ModelKind.CLASSIFIER) { "Classifier contract required" }
        require(canonicalLabels.size == 53 && canonicalLabels.distinct().size == 53 && canonicalLabels[52] == "BACK") {
            "Canonical classifier labels must contain 53 unique entries with BACK at index 52"
        }
        require(contract.labels == canonicalLabels) { "Contract label order does not match canonical labels" }
        require(scores.size == 53) { "Classifier output width must be exactly 53" }
        scores.forEachIndexed { index, score ->
            require(score.isFinite() && score in 0f..1f) { "Classifier score at index $index is not a finite probability" }
        }
        val bestIndex = scores.indices.maxWithOrNull(compareBy<Int> { scores[it] }.thenByDescending { it })!!
        val alternatives = scores.indices.asSequence().filter { it != bestIndex }
            .sortedWith(compareByDescending<Int> { scores[it] }.thenBy { it })
            .take(contract.classifier!!.topK - 1).map { index ->
                ClassifierAlternative(canonicalLabels[index], CardLabelCodec.decode(canonicalLabels[index]), scores[index])
            }.toList()
        return ClassifierPrediction(
            canonicalLabels[bestIndex], CardLabelCodec.decode(canonicalLabels[bestIndex]), scores[bestIndex], alternatives,
        )
    }
}
