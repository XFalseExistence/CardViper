package com.cardviper.app.vision.litert

import org.json.JSONArray
import org.json.JSONObject

enum class ModelKind { CLASSIFIER, DETECTOR }
enum class TensorLayout { NHWC, NCHW }
enum class TensorDataType { FLOAT32, UINT8 }
enum class ResizePolicy { STRETCH, LETTERBOX }
enum class ResizeFilter { NEAREST, BILINEAR }
enum class ChannelOrder { RGB, BGR }
enum class OutputRole { CLASSIFIER_SCORES, DETECTION_BOXES, DETECTION_SCORES, DETECTION_CLASSES, DETECTION_COUNT }
enum class BoxCoordinateOrder { XYXY, YXYX }
enum class BoxCoordinateUnits { PIXELS, NORMALIZED }
enum class NmsOwnership { EMBEDDED, HOST }

data class ModelInput(
    val name: String,
    val shape: List<Int>,
    val layout: TensorLayout,
    val dtype: TensorDataType,
) {
    val width: Int get() = if (layout == TensorLayout.NHWC) shape[2] else shape[3]
    val height: Int get() = if (layout == TensorLayout.NHWC) shape[1] else shape[2]
}

data class ModelTensor(
    val name: String,
    val role: OutputRole,
    val shape: List<Int>,
    val dtype: TensorDataType,
)

data class ImagePreprocessing(
    val resizePolicy: ResizePolicy,
    val resizeFilter: ResizeFilter,
    val channelOrder: ChannelOrder,
    val scale: Double,
    val offset: Double,
    val padValue: Int,
)

enum class ClassifierScoreSemantics { PROBABILITIES }
data class ClassifierSemantics(val topK: Int, val scoreSemantics: ClassifierScoreSemantics)
data class DetectorNms(val ownership: NmsOwnership, val scoreThreshold: Float, val iouThreshold: Float)
data class DetectorSemantics(
    val semanticClasses: List<String>,
    val cardClassIndex: Int,
    val boxesTensor: String,
    val coordinateOrder: BoxCoordinateOrder,
    val coordinateUnits: BoxCoordinateUnits,
    val scoresTensor: String,
    val classesTensor: String,
    val countTensor: String?,
    val nms: DetectorNms,
)

/** Versioned model metadata parser. It accepts test contracts and future real contracts, never supplies defaults. */
data class LiteRtModelContract(
    val schemaVersion: Int,
    val kind: ModelKind,
    val artifactSha256: String,
    val labelsSha256: String?,
    val input: ModelInput,
    val preprocessing: ImagePreprocessing,
    val outputs: List<ModelTensor>,
    val labels: List<String>,
    val classifier: ClassifierSemantics?,
    val detector: DetectorSemantics?,
) {
    companion object {
        fun parse(json: String): LiteRtModelContract = try {
            parseRoot(JSONObject(json))
        } catch (failure: IllegalArgumentException) {
            throw failure
        } catch (failure: Exception) {
            throw IllegalArgumentException("Invalid LiteRT contract JSON: ${failure.message}", failure)
        }

        private fun parseRoot(root: JSONObject): LiteRtModelContract {
            root.only("schema_version", "model_kind", "artifact", "input", "preprocessing", "outputs", "labels", "labels_sha256", "classifier", "detector")
            val version = root.requiredInt("schema_version")
            require(version == 1) { "Unsupported contract schema version: $version" }
            val kind = root.requiredEnum<ModelKind>("model_kind")
            val artifact = root.requiredObject("artifact").also { it.only("sha256") }
            val artifactHash = artifact.requiredString("sha256").also(::requireSha256)

            val inputObject = root.requiredObject("input").also { it.only("name", "shape", "layout", "dtype") }
            val input = ModelInput(
                inputObject.requiredString("name").also { require(it.isNotBlank()) { "Input name is required" } },
                inputObject.requiredIntArray("shape"),
                inputObject.requiredEnum("layout"),
                inputObject.requiredEnum("dtype"),
            )
            require(input.shape.size == 4) { "Input shape must have rank four" }
            require(input.shape.all { it > 0 }) { "Input dimensions must be positive" }
            require(input.shape[0] == 1) { "Only batch size one is supported" }
            require(input.shape[if (input.layout == TensorLayout.NHWC) 3 else 1] == 3) { "Input must have exactly three channels" }

            val prepObject = root.requiredObject("preprocessing").also { it.only("resize_policy", "resize_filter", "channel_order", "scale", "offset", "pad_value") }
            val preprocessing = ImagePreprocessing(
                prepObject.requiredEnum("resize_policy"), prepObject.requiredEnum("resize_filter"), prepObject.requiredEnum("channel_order"),
                prepObject.requiredFiniteDouble("scale"), prepObject.requiredFiniteDouble("offset"), prepObject.requiredInt("pad_value"),
            )
            require(preprocessing.padValue in 0..255) { "Padding value must be a byte" }

            val outputsArray = root.requiredArray("outputs")
            require(outputsArray.length() > 0) { "At least one output tensor is required" }
            val outputs = (0 until outputsArray.length()).map { index ->
                val output = outputsArray.getJSONObject(index).also { it.only("name", "role", "shape", "dtype") }
                ModelTensor(
                    output.requiredString("name").also { require(it.isNotBlank()) { "Output name is required" } },
                    output.requiredEnum("role"),
                    output.requiredIntArray("shape").also { shape ->
                        require(shape.isNotEmpty() && shape.all { it > 0 }) { "Output dimensions must be positive" }
                    },
                    output.requiredEnum("dtype"),
                )
            }
            require(outputs.map { it.name }.distinct().size == outputs.size) { "Output tensor names must be unique" }
            require(outputs.map { it.role }.distinct().size == outputs.size) { "Output tensor roles must be unique" }
            require(outputs.all { it.dtype == TensorDataType.FLOAT32 }) { "Only FLOAT32 outputs are supported" }

            val labels = if (root.has("labels") && !root.isNull("labels")) root.requiredArray("labels").toStringList("labels") else emptyList()
            val labelsHash = if (root.has("labels_sha256") && !root.isNull("labels_sha256")) root.requiredString("labels_sha256").also(::requireSha256) else null
            val classifier = if (kind == ModelKind.CLASSIFIER) {
                require(root.has("classifier") && !root.isNull("classifier")) { "Classifier semantics are required" }
                root.requiredObject("classifier").also { it.only("top_k", "score_semantics") }.let {
                    ClassifierSemantics(it.requiredInt("top_k"), it.requiredEnum("score_semantics"))
                }
            } else null
            val detector = if (kind == ModelKind.DETECTOR) {
                require(root.has("detector") && !root.isNull("detector")) { "Detector semantics are required" }
                parseDetector(root.requiredObject("detector"))
            } else null

            if (kind == ModelKind.CLASSIFIER) {
                require(!root.has("detector") || root.isNull("detector")) { "Classifier contract cannot declare detector semantics" }
                require(labels.size == 53) { "Classifier must declare exactly 53 labels" }
                require(labels.distinct().size == labels.size) { "Classifier labels must be unique" }
                require(labels[52] == "BACK") { "BACK must be canonical classifier index 52" }
                require(labelsHash != null) { "Classifier label SHA-256 is required" }
                labels.forEach { label ->
                    try { com.cardviper.app.vision.CardLabelCodec.decode(label) }
                    catch (failure: Exception) { throw IllegalArgumentException("Invalid classifier label: $label", failure) }
                }
                require(classifier!!.topK in 1..53) { "Classifier top_k must be between 1 and 53" }
                require(outputs.size == 1 && outputs.single().role == OutputRole.CLASSIFIER_SCORES) { "Classifier must have one score output" }
                require(outputs.single().shape.last() == 53) { "Classifier output width must be 53" }
                require(outputs.single().shape == listOf(1, 53)) { "Classifier score output must have shape [1,53]" }
                require(outputs.single().dtype == TensorDataType.FLOAT32) { "Classifier score output must be FLOAT32" }
            } else {
                require(labels.isEmpty()) { "Detector semantic classes are declared in detector, not labels" }
                require(labelsHash == null) { "Detector contracts cannot declare classifier labels hash" }
                require(classifier == null) { "Detector contract cannot declare classifier semantics" }
                validateDetectorOutputs(outputs, detector!!)
            }

            return LiteRtModelContract(version, kind, artifactHash, labelsHash, input, preprocessing, outputs, labels, classifier, detector)
        }

        private fun parseDetector(obj: JSONObject): DetectorSemantics {
            obj.only("semantic_classes", "card_class_index", "boxes", "scores", "classes", "count", "nms")
            val classes = obj.requiredArray("semantic_classes").toStringList("semantic_classes")
            require(classes == listOf("CARD")) { "Detector semantic classes must be exactly [CARD]" }
            val classIndex = obj.requiredInt("card_class_index")
            require(classIndex == 0) { "The CARD class index must be zero" }
            val boxes = obj.requiredObject("boxes").also { it.only("tensor", "coordinate_order", "coordinate_units") }
            val scores = obj.requiredObject("scores").also { it.only("tensor") }
            val detectorClasses = obj.requiredObject("classes").also { it.only("tensor") }
            val count = if (!obj.has("count") || obj.isNull("count")) null else obj.requiredObject("count").also { it.only("tensor") }.requiredString("tensor")
            val nmsObj = obj.requiredObject("nms").also { it.only("ownership", "score_threshold", "iou_threshold") }
            val nms = DetectorNms(
                nmsObj.requiredEnum("ownership"), nmsObj.requiredUnitFloat("score_threshold"), nmsObj.requiredUnitFloat("iou_threshold"),
            )
            return DetectorSemantics(
                classes,
                classIndex,
                boxes.requiredString("tensor"),
                boxes.requiredEnum("coordinate_order"),
                boxes.requiredEnum("coordinate_units"),
                scores.requiredString("tensor"),
                detectorClasses.requiredString("tensor"),
                count,
                nms,
            )
        }

        private fun validateDetectorOutputs(outputs: List<ModelTensor>, detector: DetectorSemantics) {
            val byRole = outputs.associateBy { it.role }
            require(byRole.keys.containsAll(listOf(OutputRole.DETECTION_BOXES, OutputRole.DETECTION_SCORES, OutputRole.DETECTION_CLASSES))) {
                "Detector box, score, and class output tensors are required"
            }
            require(byRole.keys.all { it in setOf(OutputRole.DETECTION_BOXES, OutputRole.DETECTION_SCORES, OutputRole.DETECTION_CLASSES, OutputRole.DETECTION_COUNT) }) {
                "Detector has an unsupported output role"
            }
            require(outputs.size == if (detector.countTensor == null) 3 else 4) { "Detector outputs do not match count tensor declaration" }
            require(byRole.getValue(OutputRole.DETECTION_BOXES).name == detector.boxesTensor) { "Box tensor declaration does not match output list" }
            require(byRole.getValue(OutputRole.DETECTION_SCORES).name == detector.scoresTensor) { "Score tensor declaration does not match output list" }
            require(byRole.getValue(OutputRole.DETECTION_CLASSES).name == detector.classesTensor) { "Class tensor declaration does not match output list" }
            require(byRole[OutputRole.DETECTION_COUNT]?.name == detector.countTensor) { "Count tensor declaration does not match output list" }
            val boxes = byRole.getValue(OutputRole.DETECTION_BOXES)
            require(boxes.shape.size == 3 && boxes.shape[0] == 1) { "Detection boxes must have shape [1,N,4]" }
            require(boxes.shape.last() == 4) { "Detection boxes must have a final dimension of four" }
            listOf(OutputRole.DETECTION_SCORES, OutputRole.DETECTION_CLASSES).forEach { role ->
                val tensor = byRole.getValue(role)
                require(tensor.shape == listOf(1, boxes.shape[1])) { "Detection tensor lengths must agree" }
            }
            byRole[OutputRole.DETECTION_COUNT]?.let { require(it.shape == listOf(1)) { "Detection count tensor must have shape [1]" } }
        }

        private fun requireSha256(value: String) {
            require(value.matches(Regex("[0-9a-fA-F]{64}"))) { "SHA-256 must contain exactly 64 hexadecimal characters" }
        }
    }
}

private inline fun <reified T : Enum<T>> JSONObject.requiredEnum(key: String): T {
    val raw = requiredString(key)
    return enumValues<T>().firstOrNull { it.name == raw }
        ?: throw IllegalArgumentException("Unsupported ${T::class.simpleName} value '$raw' for $key")
}

private fun JSONObject.only(vararg allowed: String) {
    val keys = keys()
    while (keys.hasNext()) {
        val key = keys.next()
        require(key in allowed) { "Unsupported field '$key'" }
    }
}

private fun JSONObject.requiredObject(key: String): JSONObject = get(key) as? JSONObject
    ?: throw IllegalArgumentException("$key must be an object")
private fun JSONObject.requiredArray(key: String): JSONArray = get(key) as? JSONArray
    ?: throw IllegalArgumentException("$key must be an array")
private fun JSONObject.requiredString(key: String): String = get(key) as? String
    ?: throw IllegalArgumentException("$key must be a string")
private fun JSONObject.requiredInt(key: String): Int {
    val value = get(key) as? Number ?: throw IllegalArgumentException("$key must be an integer")
    val doubleValue = value.toDouble()
    require(doubleValue.isFinite() && doubleValue % 1.0 == 0.0 && doubleValue in Int.MIN_VALUE.toDouble()..Int.MAX_VALUE.toDouble()) { "$key must be an integer" }
    return doubleValue.toInt()
}
private fun JSONObject.requiredFiniteDouble(key: String): Double {
    val value = (get(key) as? Number)?.toDouble() ?: throw IllegalArgumentException("$key must be numeric")
    require(value.isFinite()) { "$key must be finite" }
    return value
}
private fun JSONObject.requiredUnitFloat(key: String): Float {
    val value = requiredFiniteDouble(key)
    require(value in 0.0..1.0) { "$key must be within 0..1" }
    return value.toFloat()
}
private fun JSONObject.requiredIntArray(key: String): List<Int> {
    val array = requiredArray(key)
    return (0 until array.length()).map { index ->
        val number = array.get(index) as? Number ?: throw IllegalArgumentException("$key entries must be integers")
        val value = number.toDouble()
        require(value.isFinite() && value % 1.0 == 0.0 && value in Int.MIN_VALUE.toDouble()..Int.MAX_VALUE.toDouble()) { "$key entries must be integers" }
        value.toInt()
    }
}
private fun JSONArray.toStringList(field: String): List<String> = (0 until length()).map { index ->
    get(index) as? String ?: throw IllegalArgumentException("$field entries must be strings")
}
