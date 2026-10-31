package com.cardviper.app.vision.litert

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

class LiteRtGeometryParityTest {
    @Test fun letterboxTransformsMatchPythonDetectorGeometryFixtures() {
        val root = JSONObject(javaClass.getResource("/vision/python-detector-geometry-parity.TEST_ONLY.json")!!.readText())
        val cases = root.getJSONArray("cases")
        repeat(cases.length()) { index ->
            val case = cases.getJSONObject(index)
            val source = case.getJSONArray("source")
            val model = case.getJSONArray("model")
            val box = case.getJSONArray("box")
            val expected = case.getJSONArray("model_box")
            val transform = LiteRtResizeTransform(source.getInt(0), source.getInt(1), model.getInt(0), model.getInt(1), ResizePolicy.LETTERBOX)
            val actual = transform.sourceToModel(FloatBox(box.getDouble(0).toFloat(), box.getDouble(1).toFloat(), box.getDouble(2).toFloat(), box.getDouble(3).toFloat()))
            assertEquals(expected.getDouble(0).toFloat(), actual.left, 0.0001f)
            assertEquals(expected.getDouble(1).toFloat(), actual.top, 0.0001f)
            assertEquals(expected.getDouble(2).toFloat(), actual.right, 0.0001f)
            assertEquals(expected.getDouble(3).toFloat(), actual.bottom, 0.0001f)
        }
    }
}
