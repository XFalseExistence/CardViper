package com.cardviper.app.vision.litert

internal class FakeLiteRtRuntime(
    private val signatureOverride: TensorSignature? = null,
    var outputs: Map<String, FloatArray> = emptyMap(),
    var failure: Exception? = null,
) : LiteRtRuntimeFactory {
    var openCalls = 0
        private set
    var inferenceCalls = 0
        private set
    var closed = false
        private set
    var lastInput: PreprocessedImage? = null
        private set
    var modelBytesSeen: ByteArray? = null

    override fun open(modelBytes: ByteArray, contract: LiteRtModelContract): LiteRtRuntimeSession {
        openCalls++
        modelBytesSeen = modelBytes.copyOf()
        return object : LiteRtRuntimeSession {
            override val signature = signatureOverride ?: expectedSignature(contract)
            override fun run(input: PreprocessedImage): Map<String, FloatArray> {
                inferenceCalls++
                lastInput = input
                failure?.let { throw it }
                return outputs
            }
            override fun close() { closed = true }
        }
    }
}

internal data class AdapterBundleFixture(
    val bundle: LiteRtModelBundle,
    val contract: LiteRtModelContract,
    val modelBytes: ByteArray,
    val labelsBytes: ByteArray?,
)

internal fun classifierBundleFixture(): AdapterBundleFixture {
    val model = "test-only classifier bytes".toByteArray()
    val labels = (canonicalAdapterLabels().joinToString("\n") + "\n").toByteArray()
    val raw = javaClassResource("/vision/classifier.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")
        .replaceFirst("0".repeat(64), LiteRtArtifactValidator.sha256(model))
        .replaceFirst("0".repeat(64), LiteRtArtifactValidator.sha256(labels))
    val contract = LiteRtModelContract.parse(raw)
    return AdapterBundleFixture(LiteRtModelBundle(raw, model, labels), contract, model, labels)
}

internal fun detectorBundleFixture(
    ownership: NmsOwnership = NmsOwnership.HOST,
    includeCount: Boolean = true,
): AdapterBundleFixture {
    val model = "test-only detector bytes".toByteArray()
    var raw = javaClassResource("/vision/detector.TEST_ONLY.SYNTHETIC.NOT_PRODUCTION_CONTRACT.json")
    raw = raw.replace("\"ownership\": \"HOST\"", "\"ownership\": \"${ownership.name}\"")
    if (!includeCount) {
        raw = raw.replace(",\n    { \"name\": \"count\", \"role\": \"DETECTION_COUNT\", \"shape\": [1], \"dtype\": \"FLOAT32\" }", "")
            .replace("    \"count\": { \"tensor\": \"count\" },\n", "")
    }
    raw = raw.replace("1".repeat(64), LiteRtArtifactValidator.sha256(model))
    val contract = LiteRtModelContract.parse(raw)
    return AdapterBundleFixture(LiteRtModelBundle(raw, model, null), contract, model, null)
}

internal fun expectedSignature(contract: LiteRtModelContract) = TensorSignature(
    ObservedTensor(contract.input.name, contract.input.shape, contract.input.dtype),
    contract.outputs.map { ObservedTensor(it.name, it.shape, it.dtype) },
)

internal fun canonicalAdapterLabels() = listOf("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
    .flatMap { rank -> listOf("C", "D", "H", "S").map { "$rank$it" } } + "BACK"

private fun javaClassResource(path: String): String =
    LiteRtAdapterTestSupport::class.java.getResource(path)!!.readText()

private object LiteRtAdapterTestSupport
