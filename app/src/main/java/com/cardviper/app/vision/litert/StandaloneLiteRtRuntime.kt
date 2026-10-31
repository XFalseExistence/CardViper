package com.cardviper.app.vision.litert

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor

/** Pinned standalone LiteRT CPU implementation behind the existing dependency-injected seam. */
class StandaloneLiteRtRuntime(
    private val threads: Int = 1,
) : LiteRtRuntime {
    init {
        require(threads > 0) { "Thread count must be positive" }
    }

    override fun open(modelBytes: ByteArray, contract: LiteRtModelContract): LiteRtSession =
        openSession(modelBytes)

    /**
     * Runtime-only smoke entrypoint for test-only arithmetic models that intentionally have no
     * classifier/detector semantics. Production adapters must use [open], which remains behind
     * contract validation in [LiteRtRuntimeBoundary].
     */
    internal fun openForRuntimeSmoke(modelBytes: ByteArray): LiteRtSession = openSession(modelBytes)

    private fun openSession(modelBytes: ByteArray): LiteRtSession {
        require(modelBytes.isNotEmpty()) { "Model artifact is empty" }
        val model = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
        model.put(modelBytes)
        model.rewind()
        val interpreter = Interpreter(model, Interpreter.Options().setNumThreads(threads))
        try {
            interpreter.allocateTensors()
            require(interpreter.inputTensorCount == 1) { "Only single-input models are supported" }
            val input = interpreter.getInputTensor(0).observed()
            val outputs = (0 until interpreter.outputTensorCount).map { interpreter.getOutputTensor(it).observed() }
            require(outputs.isNotEmpty()) { "Model must expose at least one output tensor" }
            require(outputs.all { it.dtype == TensorDataType.FLOAT32 }) { "Only FLOAT32 model outputs are supported" }
            return Session(interpreter, input, outputs)
        } catch (failure: Throwable) {
            runCatching { interpreter.close() }
            throw failure
        }
    }

    private class Session(
        private val interpreter: Interpreter,
        input: ObservedTensor,
        private val outputs: List<ObservedTensor>,
    ) : LiteRtSession {
        override val signature = TensorSignature(input, outputs)
        private var closed = false

        @Synchronized
        override fun run(input: PreprocessedImage): Map<String, FloatArray> {
            check(!closed) { "LiteRT session is closed" }
            require(input.shape == signature.input.shape) { "Inference input shape does not match runtime tensor" }
            require(input.dtype == signature.input.dtype) { "Inference input dtype does not match runtime tensor" }

            val inputBuffer = when (input.dtype) {
                TensorDataType.FLOAT32 -> {
                    val values = input.floatValues ?: throw IllegalArgumentException("FLOAT32 input values are required")
                    require(values.size == elementCount(input.shape)) { "Inference FLOAT32 input length does not match shape" }
                    ByteBuffer.allocateDirect(values.size * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())
                        .also { buffer -> values.forEach { buffer.putFloat(it) }; buffer.rewind() }
                }
                TensorDataType.UINT8 -> {
                    val values = input.byteValues ?: throw IllegalArgumentException("UINT8 input values are required")
                    require(values.size == elementCount(input.shape)) { "Inference UINT8 input length does not match shape" }
                    ByteBuffer.allocateDirect(values.size).order(ByteOrder.nativeOrder())
                        .also { buffer -> buffer.put(values); buffer.rewind() }
                }
            }

            val outputBuffers = outputs.indices.associateWith { index ->
                ByteBuffer.allocateDirect(interpreter.getOutputTensor(index).numBytes()).order(ByteOrder.nativeOrder())
            }.toMutableMap<Int, Any>()
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputBuffers)
            return outputs.indices.associate { index ->
                val buffer = outputBuffers.getValue(index) as ByteBuffer
                val values = FloatArray(elementCount(outputs[index].shape))
                buffer.rewind()
                buffer.asFloatBuffer().get(values)
                outputs[index].name to values
            }
        }

        @Synchronized
        override fun close() {
            if (!closed) {
                closed = true
                interpreter.close()
            }
        }
    }

    private companion object {
        fun Tensor.observed(): ObservedTensor = ObservedTensor(
            name = name(),
            shape = shape().toList(),
            dtype = when (dataType()) {
                DataType.FLOAT32 -> TensorDataType.FLOAT32
                DataType.UINT8 -> TensorDataType.UINT8
                else -> throw IllegalArgumentException("Unsupported LiteRT tensor dtype: ${dataType()}")
            },
        )

        fun elementCount(shape: List<Int>): Int = shape.fold(1) { total, dimension ->
            require(dimension > 0) { "Runtime tensor dimensions must be positive" }
            Math.multiplyExact(total, dimension)
        }
    }
}
