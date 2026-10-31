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
            require(outputs.map { it.name }.toSet().size == outputs.size) { "Output tensor names must be unique" }
            require(outputs.all { it.dtype == TensorDataType.FLOAT32 }) { "Only FLOAT32 model outputs are supported" }
            outputs.forEach { require(StandaloneLiteRtBufferCodec.elementCount(it.shape) > 0) { "Output tensor shape must be non-empty" } }
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
            val inputBuffer = StandaloneLiteRtBufferCodec.encodeInput(input, signature.input)

            val outputBuffers = outputs.indices.associateWith { index ->
                ByteBuffer.allocateDirect(interpreter.getOutputTensor(index).numBytes()).order(ByteOrder.nativeOrder())
            }.toMutableMap<Int, Any>()
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputBuffers)
            return outputs.indices.associate { index ->
                val buffer = outputBuffers.getValue(index) as ByteBuffer
                outputs[index].name to StandaloneLiteRtBufferCodec.decodeOutput(buffer, outputs[index])
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

    }
}

/** Pure tensor marshalling shared by the Android runtime and JVM boundary tests. */
internal object StandaloneLiteRtBufferCodec {
    fun encodeInput(input: PreprocessedImage, observed: ObservedTensor): ByteBuffer {
        require(input.shape == observed.shape) { "Inference input shape does not match runtime tensor" }
        require(input.dtype == observed.dtype) { "Inference input dtype does not match runtime tensor" }
        val count = elementCount(observed.shape)
        return when (observed.dtype) {
            TensorDataType.FLOAT32 -> {
                val values = input.floatValues ?: throw IllegalArgumentException("FLOAT32 input values are required")
                require(values.size == count) { "Inference FLOAT32 input length does not match shape" }
                ByteBuffer.allocateDirect(Math.multiplyExact(count, Float.SIZE_BYTES)).order(ByteOrder.nativeOrder())
                    .also { buffer -> values.forEach { buffer.putFloat(it) }; buffer.rewind() }
            }
            TensorDataType.UINT8 -> {
                val values = input.byteValues ?: throw IllegalArgumentException("UINT8 input values are required")
                require(values.size == count) { "Inference UINT8 input length does not match shape" }
                ByteBuffer.allocateDirect(count).order(ByteOrder.nativeOrder())
                    .also { buffer -> buffer.put(values); buffer.rewind() }
            }
        }
    }

    fun decodeOutput(buffer: ByteBuffer, observed: ObservedTensor): FloatArray {
        require(observed.dtype == TensorDataType.FLOAT32) { "Only FLOAT32 model outputs are supported" }
        val byteCount = Math.multiplyExact(elementCount(observed.shape), Float.SIZE_BYTES)
        require(buffer.capacity() == byteCount) { "Output buffer byte length does not match tensor shape" }
        val values = FloatArray(byteCount / Float.SIZE_BYTES)
        buffer.duplicate().order(ByteOrder.nativeOrder()).also { it.rewind() }.asFloatBuffer().get(values)
        return values
    }

    fun elementCount(shape: List<Int>): Int = shape.fold(1) { total, dimension ->
        require(dimension > 0) { "Runtime tensor dimensions must be positive" }
        Math.multiplyExact(total, dimension)
    }
}
