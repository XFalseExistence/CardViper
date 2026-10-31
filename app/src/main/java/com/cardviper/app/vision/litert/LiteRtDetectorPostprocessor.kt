package com.cardviper.app.vision.litert

import com.cardviper.app.vision.CardCandidate
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** Raw logical detector tensors after LiteRT extraction, before source-coordinate reversal. */
data class DetectorOutputs(
    val boxes: FloatArray,
    val scores: FloatArray,
    val classes: FloatArray,
    val count: Float? = null,
)

object LiteRtDetectorPostprocessor {
    fun process(
        outputs: DetectorOutputs,
        contract: LiteRtModelContract.Detector,
        transform: ModelTransform,
    ): List<CardCandidate> {
        require(transform.modelWidth == contract.input.width && transform.modelHeight == contract.input.height) {
            "Detector transform model size does not match contract input"
        }
        require(outputs.boxes.size % 4 == 0) { "Detector boxes length must be divisible by four" }
        val capacity = outputs.boxes.size / 4
        require(outputs.scores.size == capacity && outputs.classes.size == capacity) {
            "Detector box/score/class tensor capacities must match"
        }
        val requestedCount = outputs.count?.let { count ->
            require(count.isFinite() && count >= 0f && abs(count - count.toInt()) < 1e-4f) {
                "Detector count must be a finite nonnegative integer value"
            }
            count.toInt().coerceAtMost(capacity)
        } ?: capacity

        val decoded = buildList {
            for (index in 0 until requestedCount) {
                val score = outputs.scores[index]
                val rawClass = outputs.classes[index]
                require(score.isFinite()) { "Detector score[$index] must be finite" }
                require(score in 0f..1f) { "Detector score[$index] must be in 0..1" }
                require(rawClass.isFinite() && abs(rawClass - rawClass.toInt()) < 1e-4f) {
                    "Detector class[$index] must be an integer-valued finite float"
                }
                require(rawClass.toInt() == contract.output.cardClassIndex) {
                    "Detector produced unexpected class ${rawClass.toInt()}"
                }
                if (score < contract.output.scoreThreshold) continue
                val base = index * 4
                val raw = outputs.boxes.copyOfRange(base, base + 4)
                require(raw.all { it.isFinite() }) { "Detector box[$index] must be finite" }
                var modelRect = when (contract.output.boxOrder) {
                    BoxOrder.YXYX -> FloatRect(raw[1], raw[0], raw[3], raw[2])
                    BoxOrder.XYXY -> FloatRect(raw[0], raw[1], raw[2], raw[3])
                }
                if (contract.output.boxUnits == BoxUnits.NORMALIZED) {
                    modelRect = FloatRect(
                        modelRect.left * transform.modelWidth,
                        modelRect.top * transform.modelHeight,
                        modelRect.right * transform.modelWidth,
                        modelRect.bottom * transform.modelHeight,
                    )
                }
                val source = transform.modelToSource(modelRect)
                val clamped = FloatRect(
                    left = source.left.coerceIn(0f, transform.sourceWidth.toFloat()),
                    top = source.top.coerceIn(0f, transform.sourceHeight.toFloat()),
                    right = source.right.coerceIn(0f, transform.sourceWidth.toFloat()),
                    bottom = source.bottom.coerceIn(0f, transform.sourceHeight.toFloat()),
                )
                if (clamped.right <= clamped.left || clamped.bottom <= clamped.top) continue
                add(DecodedBox(clamped, score))
            }
        }

        val selected = when (contract.output.nmsOwnership) {
            NmsOwnership.EMBEDDED -> decoded
            NmsOwnership.HOST -> hostNms(decoded, contract.output.nmsThreshold)
        }
        return selected.map { box ->
            CardCandidate(
                x = box.rect.left,
                y = box.rect.top,
                width = box.rect.right - box.rect.left,
                height = box.rect.bottom - box.rect.top,
                confidence = box.score,
            )
        }
    }

    private data class DecodedBox(val rect: FloatRect, val score: Float)

    private fun hostNms(boxes: List<DecodedBox>, threshold: Float): List<DecodedBox> {
        val remaining = boxes.sortedWith(
            compareByDescending<DecodedBox> { it.score }
                .thenBy { it.rect.top }
                .thenBy { it.rect.left },
        ).toMutableList()
        val kept = mutableListOf<DecodedBox>()
        while (remaining.isNotEmpty()) {
            val next = remaining.removeAt(0)
            kept += next
            remaining.removeAll { iou(next.rect, it.rect) > threshold }
        }
        return kept
    }

    private fun iou(a: FloatRect, b: FloatRect): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val areaA = max(0f, a.right - a.left) * max(0f, a.bottom - a.top)
        val areaB = max(0f, b.right - b.left) * max(0f, b.bottom - b.top)
        val union = areaA + areaB - intersection
        return if (union <= 0f) 0f else intersection / union
    }
}
