package com.cardviper.app.vision.litert

internal fun resourceText(name: String): String = requireNotNull(
    LiteRtModelContract::class.java.getResource("/vision/litert/$name"),
) { "Missing test resource $name" }.readText()

internal fun classifierContract(): LiteRtModelContract.Classifier =
    LiteRtModelContract.parse(resourceText("classifier-contract.json")) as LiteRtModelContract.Classifier

internal fun detectorEmbeddedContract(): LiteRtModelContract.Detector =
    LiteRtModelContract.parse(resourceText("detector-embedded-contract.json")) as LiteRtModelContract.Detector

internal fun detectorHostContract(): LiteRtModelContract.Detector =
    LiteRtModelContract.parse(resourceText("detector-host-contract.json")) as LiteRtModelContract.Detector
