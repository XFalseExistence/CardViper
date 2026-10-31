package com.cardviper.app.vision.litert

import com.cardviper.app.vision.CardLabelCodec

private val SHA256_RE = Regex("^[0-9a-f]{64}$")

enum class LiteRtModelKind { CLASSIFIER, DETECTOR }
enum class TensorDType { FLOAT32, UINT8 }
enum class TensorLayout { NHWC }
enum class ChannelOrder { RGB, BGR }
enum class ResamplingMode { NEAREST, BILINEAR }
enum class ResizePolicy { STRETCH, LETTERBOX }
enum class BoxOrder { YXYX, XYXY }
enum class BoxUnits { NORMALIZED, MODEL_PIXELS }
enum class NmsOwnership { EMBEDDED, HOST }

data class ArtifactContract(
    val sha256: String,
    val sizeBytes: Long,
)

data class InputTensorContract(
    val width: Int,
    val height: Int,
    val channels: Int,
    val dtype: TensorDType,
    val layout: TensorLayout,
)

data class NormalizationContract(
    /** Model value = unsigned channel byte * multiplier + offset. */
    val multiplier: Float,
    val offset: Float,
)

data class PreprocessingContract(
    val channelOrder: ChannelOrder,
    val normalization: NormalizationContract?,
    val paddingValue: Int,
    val resampling: ResamplingMode,
)

data class ResizeContract(val policy: ResizePolicy)

data class TensorRoleContract(val index: Int)

data class DetectorOutputContract(
    val boxes: TensorRoleContract,
    val scores: TensorRoleContract,
    val classes: TensorRoleContract,
    val count: TensorRoleContract?,
    val boxOrder: BoxOrder,
    val boxUnits: BoxUnits,
    val nmsOwnership: NmsOwnership,
    val scoreThreshold: Float,
    val nmsThreshold: Float,
    val cardClassIndex: Int,
)

sealed interface LiteRtModelContract {
    val schemaVersion: Int
    val artifact: ArtifactContract
    val input: InputTensorContract
    val preprocessing: PreprocessingContract
    val resize: ResizeContract

    data class Classifier(
        override val schemaVersion: Int,
        override val artifact: ArtifactContract,
        override val input: InputTensorContract,
        override val preprocessing: PreprocessingContract,
        override val resize: ResizeContract,
        val labels: List<String>,
        val labelsSha256: String,
        val scoreDtype: TensorDType,
    ) : LiteRtModelContract

    data class Detector(
        override val schemaVersion: Int,
        override val artifact: ArtifactContract,
        override val input: InputTensorContract,
        override val preprocessing: PreprocessingContract,
        override val resize: ResizeContract,
        val semanticClasses: List<String>,
        val output: DetectorOutputContract,
    ) : LiteRtModelContract

    companion object {
        val canonicalClassifierLabels: List<String> = buildList {
            val ranks = listOf("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
            val suits = listOf("C", "D", "H", "S")
            for (rank in ranks) for (suit in suits) add(rank + suit)
            add("BACK")
        }

        fun parse(json: String): LiteRtModelContract {
            val root = StrictJson.parse(json).asObject("contract")
            val schema = root.requiredInt("schema_version")
            require(schema == 1) { "Unsupported contract schema_version: $schema" }
            return when (root.requiredEnum<LiteRtModelKind>("model_kind")) {
                LiteRtModelKind.CLASSIFIER -> parseClassifier(schema, root)
                LiteRtModelKind.DETECTOR -> parseDetector(schema, root)
            }
        }

        private fun parseClassifier(schema: Int, root: JsonObject): Classifier {
            val common = parseCommon(root)
            val output = root.requiredObject("output")
            val labels = output.requiredStringList("labels")
            require(labels.size == 53) { "Classifier contract must contain exactly 53 labels" }
            require(labels.toSet().size == labels.size) { "Classifier output labels must be unique" }
            require(labels[52] == "BACK") { "BACK must occupy canonical classifier index 52" }
            require(labels == canonicalClassifierLabels) { "Classifier output label order is not canonical" }
            labels.forEach { label ->
                require(CardLabelCodec.encode(CardLabelCodec.decode(label)) == label) {
                    "Classifier label is not canonical: $label"
                }
            }
            val labelsSha = output.requiredString("labels_sha256")
            require(SHA256_RE.matches(labelsSha)) { "labels_sha256 must be lowercase SHA-256 hex" }
            return Classifier(
                schemaVersion = schema,
                artifact = common.artifact,
                input = common.input,
                preprocessing = common.preprocessing,
                resize = common.resize,
                labels = labels,
                labelsSha256 = labelsSha,
                scoreDtype = output.requiredEnum("score_dtype"),
            )
        }

        private fun parseDetector(schema: Int, root: JsonObject): Detector {
            val common = parseCommon(root)
            val semantic = root.requiredStringList("semantic_classes")
            require(semantic == listOf("CARD")) { "Detector semantic_classes must be exactly [CARD]" }
            val output = root.requiredObject("output")
            val countValue = output["count"]
            val count = when (countValue) {
                null, JsonNull -> null
                else -> parseRole(countValue.asObject("output.count"), "count")
            }
            val scoreThreshold = output.requiredFiniteFloat("score_threshold")
            val nmsThreshold = output.requiredFiniteFloat("nms_threshold")
            require(scoreThreshold in 0f..1f) { "score_threshold must be in 0..1" }
            require(nmsThreshold in 0f..1f) { "nms_threshold must be in 0..1" }
            val detectorOutput = DetectorOutputContract(
                boxes = parseRole(output.requiredObject("boxes"), "boxes"),
                scores = parseRole(output.requiredObject("scores"), "scores"),
                classes = parseRole(output.requiredObject("classes"), "classes"),
                count = count,
                boxOrder = output.requiredEnum("box_order"),
                boxUnits = output.requiredEnum("box_units"),
                nmsOwnership = output.requiredEnum("nms_ownership"),
                scoreThreshold = scoreThreshold,
                nmsThreshold = nmsThreshold,
                cardClassIndex = output.requiredInt("card_class_index").also {
                    require(it >= 0) { "card_class_index cannot be negative" }
                },
            )
            val indices = listOfNotNull(
                detectorOutput.boxes.index,
                detectorOutput.scores.index,
                detectorOutput.classes.index,
                detectorOutput.count?.index,
            )
            require(indices.toSet().size == indices.size) { "Detector tensor role indices must be unique" }
            return Detector(
                schemaVersion = schema,
                artifact = common.artifact,
                input = common.input,
                preprocessing = common.preprocessing,
                resize = common.resize,
                semanticClasses = semantic,
                output = detectorOutput,
            )
        }

        private data class Common(
            val artifact: ArtifactContract,
            val input: InputTensorContract,
            val preprocessing: PreprocessingContract,
            val resize: ResizeContract,
        )

        private fun parseCommon(root: JsonObject): Common {
            val artifactObject = root.requiredObject("artifact")
            val sha = artifactObject.requiredString("sha256")
            require(SHA256_RE.matches(sha)) { "artifact sha256 must be lowercase SHA-256 hex" }
            val size = artifactObject.requiredLong("size_bytes")
            require(size > 0L) { "artifact size_bytes must be positive" }

            val inputObject = root.requiredObject("input")
            val width = inputObject.requiredInt("width")
            val height = inputObject.requiredInt("height")
            val channels = inputObject.requiredInt("channels")
            require(width > 0 && height > 0 && channels > 0) { "Input tensor dimensions must be positive" }
            val elementCount = width.toLong() * height.toLong() * channels.toLong()
            require(elementCount <= Int.MAX_VALUE / Float.SIZE_BYTES) { "Input tensor is too large" }
            require(channels == 3) { "CardViper vision input requires exactly three channels" }
            val input = InputTensorContract(
                width = width,
                height = height,
                channels = channels,
                dtype = inputObject.requiredEnum("dtype"),
                layout = inputObject.requiredEnum("layout"),
            )

            val preprocessingObject = root.requiredObject("preprocessing")
            val paddingValue = preprocessingObject.requiredInt("padding_value")
            require(paddingValue in 0..255) { "padding_value must be an unsigned channel byte" }
            val normalizationValue = preprocessingObject["normalization"]
            val normalization = when (normalizationValue) {
                null, JsonNull -> null
                else -> normalizationValue.asObject("preprocessing.normalization").let { obj ->
                    NormalizationContract(
                        multiplier = obj.requiredFiniteFloat("multiplier"),
                        offset = obj.requiredFiniteFloat("offset"),
                    )
                }
            }
            when (input.dtype) {
                TensorDType.FLOAT32 -> require(normalization != null) {
                    "FLOAT32 input requires an explicit normalization contract"
                }
                TensorDType.UINT8 -> require(normalization == null) {
                    "UINT8 input must not invent float normalization"
                }
            }
            val preprocessing = PreprocessingContract(
                channelOrder = preprocessingObject.requiredEnum("channel_order"),
                normalization = normalization,
                paddingValue = paddingValue,
                resampling = preprocessingObject.requiredEnum("resampling"),
            )
            val resize = ResizeContract(root.requiredObject("resize").requiredEnum("policy"))
            return Common(ArtifactContract(sha, size), input, preprocessing, resize)
        }

        private fun parseRole(obj: JsonObject, name: String): TensorRoleContract {
            val index = obj.requiredInt("index")
            require(index >= 0) { "$name tensor index cannot be negative" }
            return TensorRoleContract(index)
        }
    }
}

private sealed interface JsonValue
private data class JsonObject(val values: Map<String, JsonValue>) : JsonValue {
    operator fun get(key: String): JsonValue? = values[key]
}
private data class JsonArray(val values: List<JsonValue>) : JsonValue
private data class JsonString(val value: String) : JsonValue
private data class JsonNumber(val token: String) : JsonValue
private data class JsonBoolean(val value: Boolean) : JsonValue
private data object JsonNull : JsonValue

private fun JsonValue.asObject(context: String): JsonObject = this as? JsonObject
    ?: throw IllegalArgumentException("$context must be a JSON object")

private fun JsonObject.requiredValue(key: String): JsonValue = values[key]
    ?: throw IllegalArgumentException("Missing required contract field: $key")

private fun JsonObject.requiredObject(key: String): JsonObject = requiredValue(key).asObject(key)

private fun JsonObject.requiredString(key: String): String =
    (requiredValue(key) as? JsonString)?.value
        ?: throw IllegalArgumentException("$key must be a string")

private fun JsonObject.requiredStringList(key: String): List<String> {
    val array = requiredValue(key) as? JsonArray
        ?: throw IllegalArgumentException("$key must be an array")
    return array.values.mapIndexed { index, value ->
        (value as? JsonString)?.value
            ?: throw IllegalArgumentException("$key[$index] must be a string")
    }
}

private fun JsonObject.requiredInt(key: String): Int {
    val token = (requiredValue(key) as? JsonNumber)?.token
        ?: throw IllegalArgumentException("$key must be an integer")
    require(!token.contains('.') && !token.contains('e', ignoreCase = true)) { "$key must be an integer" }
    return token.toIntOrNull() ?: throw IllegalArgumentException("$key is outside Int range")
}

private fun JsonObject.requiredLong(key: String): Long {
    val token = (requiredValue(key) as? JsonNumber)?.token
        ?: throw IllegalArgumentException("$key must be an integer")
    require(!token.contains('.') && !token.contains('e', ignoreCase = true)) { "$key must be an integer" }
    return token.toLongOrNull() ?: throw IllegalArgumentException("$key is outside Long range")
}

private fun JsonObject.requiredFiniteFloat(key: String): Float {
    val token = (requiredValue(key) as? JsonNumber)?.token
        ?: throw IllegalArgumentException("$key must be a number")
    val result = token.toFloatOrNull() ?: throw IllegalArgumentException("$key is not a float")
    require(result.isFinite()) { "$key must be finite" }
    return result
}

private inline fun <reified T : Enum<T>> JsonObject.requiredEnum(key: String): T {
    val raw = requiredString(key)
    return enumValues<T>().firstOrNull { it.name == raw }
        ?: throw IllegalArgumentException("Unsupported $key value: $raw")
}

private object StrictJson {
    fun parse(source: String): JsonValue {
        val parser = Parser(source)
        val value = parser.parseValue()
        parser.skipWhitespace()
        require(parser.atEnd()) { "Unexpected trailing JSON content" }
        return value
    }

    private class Parser(private val source: String) {
        private var at = 0
        fun atEnd(): Boolean = at == source.length
        fun skipWhitespace() {
            while (at < source.length && source[at].isWhitespace()) at++
        }

        fun parseValue(): JsonValue {
            skipWhitespace()
            require(at < source.length) { "Unexpected end of JSON" }
            return when (source[at]) {
                '{' -> parseObject()
                '[' -> parseArray()
                '"' -> JsonString(parseString())
                't' -> { expectLiteral("true"); JsonBoolean(true) }
                'f' -> { expectLiteral("false"); JsonBoolean(false) }
                'n' -> { expectLiteral("null"); JsonNull }
                '-', in '0'..'9' -> JsonNumber(parseNumber())
                else -> throw IllegalArgumentException("Unexpected JSON token at offset $at")
            }
        }

        private fun parseObject(): JsonObject {
            expect('{')
            skipWhitespace()
            val result = linkedMapOf<String, JsonValue>()
            if (peek('}')) { at++; return JsonObject(result) }
            while (true) {
                skipWhitespace()
                require(peek('"')) { "Object key must be a JSON string" }
                val key = parseString()
                require(key !in result) { "Duplicate JSON key: $key" }
                skipWhitespace()
                expect(':')
                result[key] = parseValue()
                skipWhitespace()
                when {
                    peek(',') -> at++
                    peek('}') -> { at++; return JsonObject(result) }
                    else -> throw IllegalArgumentException("Expected ',' or '}' at offset $at")
                }
            }
        }

        private fun parseArray(): JsonArray {
            expect('[')
            skipWhitespace()
            val result = mutableListOf<JsonValue>()
            if (peek(']')) { at++; return JsonArray(result) }
            while (true) {
                result += parseValue()
                skipWhitespace()
                when {
                    peek(',') -> at++
                    peek(']') -> { at++; return JsonArray(result) }
                    else -> throw IllegalArgumentException("Expected ',' or ']' at offset $at")
                }
            }
        }

        private fun parseString(): String {
            expect('"')
            val result = StringBuilder()
            val backslash = 92.toChar()
            while (at < source.length) {
                val ch = source[at++]
                when (ch) {
                    '"' -> return result.toString()
                    backslash -> {
                        require(at < source.length) { "Unterminated JSON escape" }
                        when (val escaped = source[at++]) {
                            '"', '/', backslash -> result.append(escaped)
                            'b' -> result.append(8.toChar())
                            'f' -> result.append(12.toChar())
                            'n' -> result.append(10.toChar())
                            'r' -> result.append(13.toChar())
                            't' -> result.append(9.toChar())
                            'u' -> {
                                require(at + 4 <= source.length) { "Incomplete Unicode escape" }
                                val code = source.substring(at, at + 4).toIntOrNull(16)
                                    ?: throw IllegalArgumentException("Invalid Unicode escape")
                                result.append(code.toChar())
                                at += 4
                            }
                            else -> throw IllegalArgumentException(
                                "Unsupported JSON escape code: " + escaped.code,
                            )
                        }
                    }
                    else -> {
                        require(ch.code >= 0x20) { "Control character in JSON string" }
                        result.append(ch)
                    }
                }
            }
            throw IllegalArgumentException("Unterminated JSON string")
        }

        private fun parseNumber(): String {
            val start = at
            if (peek('-')) at++
            require(at < source.length) { "Invalid JSON number" }
            if (peek('0')) {
                at++
            } else {
                require(source[at] in '1'..'9') { "Invalid JSON number" }
                while (at < source.length && source[at].isDigit()) at++
            }
            if (peek('.')) {
                at++
                val fractionStart = at
                while (at < source.length && source[at].isDigit()) at++
                require(at > fractionStart) { "Invalid JSON fraction" }
            }
            if (at < source.length && (source[at] == 'e' || source[at] == 'E')) {
                at++
                if (peek('+') || peek('-')) at++
                val exponentStart = at
                while (at < source.length && source[at].isDigit()) at++
                require(at > exponentStart) { "Invalid JSON exponent" }
            }
            return source.substring(start, at)
        }

        private fun expectLiteral(literal: String) {
            require(source.startsWith(literal, at)) { "Expected $literal at offset $at" }
            at += literal.length
        }

        private fun expect(ch: Char) {
            require(peek(ch)) { "Expected '$ch' at offset $at" }
            at++
        }

        private fun peek(ch: Char): Boolean = at < source.length && source[at] == ch
    }
}
