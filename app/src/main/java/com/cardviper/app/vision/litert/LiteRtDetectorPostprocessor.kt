package com.cardviper.app.vision.litert

import kotlin.math.max
import kotlin.math.min

data class DetectedCardBox(val box: FloatBox, val confidence: Float, val classIndex: Int)

/** Converts the explicitly declared detector outputs to source pixels. It does not guess export semantics. */
object LiteRtDetectorPostprocessor {
    fun process(
        contract: LiteRtModelContract,
        outputTensors: Map<String, FloatArray>,
        sourceWidth: Int,
        sourceHeight: Int,
    ): List<DetectedCardBox> {
        require(contract.kind == ModelKind.DETECTOR) { "Detector contract required" }
        require(sourceWidth > 0 && sourceHeight > 0) { "Source dimensions must be positive" }
        val semantics = contract.detector!!
        val expectedNames = contract.outputs.map { it.name }.toSet()
        require(outputTensors.keys == expectedNames) { "Output tensor names do not match detector contract" }
        val boxes = outputTensors.getValue(semantics.boxesTensor)
        val scores = outputTensors.getValue(semantics.scoresTensor)
        val classes = outputTensors.getValue(semantics.classesTensor)
        val boxTensor = contract.outputs.single { it.role == OutputRole.DETECTION_BOXES }
        val capacity = boxTensor.shape[boxTensor.shape.lastIndex - 1]
        require(boxes.size == capacity * 4 && scores.size == capacity && classes.size == capacity) {
            "Detection tensor lengths do not match contract capacity $capacity"
        }
        val count = semantics.countTensor?.let { name ->
            val values = outputTensors.getValue(name)
            require(values.size == 1) { "Detection count tensor must contain one value" }
            val raw = values[0]
            require(raw.isFinite() && raw % 1f == 0f && raw in 0f..capacity.toFloat()) { "Detection count is invalid" }
            raw.toInt()
        } ?: capacity
        val transform = LiteRtResizeTransform(sourceWidth, sourceHeight, contract.input.width, contract.input.height, contract.preprocessing.resizePolicy)
        val candidates = ArrayList<Pair<Int, DetectedCardBox>>(count)
        for (index in 0 until count) {
            val score = scores[index]
            require(score.isFinite() && score in 0f..1f) { "Detection score at $index must be a finite probability" }
            val rawClass = classes[index]
            require(rawClass.isFinite() && rawClass % 1f == 0f && rawClass.toInt() == semantics.cardClassIndex) {
                "Detection class at $index is not the declared CARD class"
            }
            if (score < semantics.nms.scoreThreshold) continue

            val offset = index * 4
            val raw = boxes.copyOfRange(offset, offset + 4)
            require(raw.all(Float::isFinite)) { "Detection box at $index contains nonfinite coordinates" }
            val x1: Float
            val y1: Float
            val x2: Float
            val y2: Float
            when (semantics.coordinateOrder) {
                BoxCoordinateOrder.XYXY -> { x1 = raw[0]; y1 = raw[1]; x2 = raw[2]; y2 = raw[3] }
                BoxCoordinateOrder.YXYX -> { y1 = raw[0]; x1 = raw[1]; y2 = raw[2]; x2 = raw[3] }
            }
            val unitScaleX = if (semantics.coordinateUnits == BoxCoordinateUnits.NORMALIZED) contract.input.width.toFloat() else 1f
            val unitScaleY = if (semantics.coordinateUnits == BoxCoordinateUnits.NORMALIZED) contract.input.height.toFloat() else 1f
            if (x2 <= x1 || y2 <= y1) continue
            val sourceBox = transform.modelToSource(
                FloatBox(x1 * unitScaleX, y1 * unitScaleY, x2 * unitScaleX, y2 * unitScaleY), clamp = true,
            )
            if (sourceBox.width <= 0f || sourceBox.height <= 0f) continue
            candidates += index to DetectedCardBox(sourceBox, score, semantics.cardClassIndex)
        }
        return when (semantics.nms.ownership) {
            NmsOwnership.EMBEDDED -> candidates.map { it.second }
            NmsOwnership.HOST -> hostNms(candidates, semantics.nms.iouThreshold)
        }
    }

    private fun hostNms(candidates: List<Pair<Int, DetectedCardBox>>, threshold: Float): List<DetectedCardBox> {
        val ordered = candidates.sortedWith(compareByDescending<Pair<Int, DetectedCardBox>> { it.second.confidence }.thenBy { it.first })
        val kept = ArrayList<DetectedCardBox>()
        ordered.forEach { (_, candidate) ->
            if (kept.none { intersectionOverUnion(it.box, candidate.box) > threshold }) kept += candidate
        }
        return kept
    }

    private fun intersectionOverUnion(a: FloatBox, b: FloatBox): Float {
        val width = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val height = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val intersection = width * height
        val union = a.width * a.height + b.width * b.height - intersection
        return if (union <= 0f) 0f else intersection / union
    }
}
