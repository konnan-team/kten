package eu.redbean.kten.api.tensor.operations

import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.nn.ConvolutionOperation
import eu.redbean.kten.api.tensor.serialization.CommonSerializableTensorDescriptor
import eu.redbean.kten.api.tensor.serialization.SerializableTensorData
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import java.io.Closeable

interface TensorOperations<RAW_TYPE : AbstractRawTensor<*>>: BasicTensorOperations {

    fun createRaw(shape: List<Int>, init: (Int) -> Float): RAW_TYPE

    /**
     * Creates a new raw tensor, without initialization (if the platform supports it, otherwise it will initialize the tensor with zeros)
     */
    fun createRaw(shape: List<Int>): RAW_TYPE

    fun createRawFill(shape: List<Int>, constant: Float): RAW_TYPE {
        return createRaw(shape) { constant }
    }

    fun release(vararg rawTensors: AbstractRawTensor<Any>)

    fun zerosLike(rawTensor: RAW_TYPE): RAW_TYPE

    fun zeroOut(rawTensor: RAW_TYPE)

    fun incrementRef(rawTensor: RAW_TYPE)

    @Suppress("UNCHECKED_CAST")
    fun markSurviveGC(rawTensor: AbstractRawTensor<Any>) {
        incrementRef(rawTensor as RAW_TYPE)
    }

    fun markReleasableInGC(rawTensor: AbstractRawTensor<Any>) {
        release(rawTensor)
    }

    fun pow(constant: Float, rawTensor: RAW_TYPE): RAW_TYPE

    fun concat(axis: Int, inputs: List<RAW_TYPE>): RAW_TYPE

    fun gemm(
        addMatrix: RAW_TYPE,
        matrix1: RAW_TYPE,
        matrix2: RAW_TYPE,
        alpha: Float = 1f,
        beta: Float = 1f,
        transposeFirst: Boolean = false,
        transposeSecond: Boolean = false
    ): RAW_TYPE

    fun gemv(
        addVector: RAW_TYPE,
        matrix: RAW_TYPE,
        vector: RAW_TYPE,
        alpha: Float = 1f,
        beta: Float = 1f,
        transposeMatrix: Boolean = false
    ): RAW_TYPE

    /**
     * Batched version of GEMM where all tensors contain a single leading batch dimension with the same size in all tensors
     */
    fun gemmBatched(
        addTensor: RAW_TYPE,
        tensor1: RAW_TYPE,
        tensor2: RAW_TYPE,
        alpha: Float = 1f,
        beta: Float = 1f,
        transposeFirst: Boolean = false,
        transposeSecond: Boolean = false
    ): RAW_TYPE

    fun mm(matrix1: RAW_TYPE, matrix2: RAW_TYPE, transposeFirst: Boolean = false, transposeSecond: Boolean = false): RAW_TYPE {
        val addMatrixShape = listOf(
            if (transposeFirst) matrix1.shape[1] else matrix1.shape[0],
            if (transposeSecond) matrix2.shape[0] else matrix2.shape[1]
        )
        val addMatrix = createRawFill(addMatrixShape,0.0f)
        return gemm(addMatrix, matrix1, matrix2, 1f, 0f, transposeFirst, transposeSecond)
    }

    fun ger(vector1: RAW_TYPE, vector2: RAW_TYPE, matrix: RAW_TYPE, alpha: Float = 1f): RAW_TYPE

    fun outer(vector1: RAW_TYPE, vector2: RAW_TYPE): RAW_TYPE {
        val addMatrix = createRawFill(listOf(vector1.shape[0], vector2.shape[0]), 0.0f)
        return ger(vector1, vector2, addMatrix)
    }

    fun mv(matrix: RAW_TYPE, vector: RAW_TYPE, transposeMatrix: Boolean = false): RAW_TYPE {
        val addVectorShape = if (transposeMatrix) listOf(matrix.shape[1]) else listOf(matrix.shape[0])
        val addVector = createRawFill(addVectorShape, 0.0f)
        return gemv(addVector, matrix, vector, 1f, 0f, transposeMatrix)
    }

    fun bmm(tensor1: RAW_TYPE, tensor2: RAW_TYPE, transposeFirst: Boolean = false, transposeSecond: Boolean = false): RAW_TYPE {
        val addTensorShape = listOf(
            tensor1.shape[0],
            if (transposeFirst) tensor1.shape[2] else tensor1.shape[1],
            if (transposeSecond) tensor2.shape[1] else tensor2.shape[2]
        )
        val addTensor = createRawFill(addTensorShape, 0.0f)
        return gemmBatched(addTensor, tensor1, tensor2, 1f, 0f, transposeFirst, transposeSecond)
    }

    fun spatialConvolution(kernel: List<Int>, padding: List<Int>, stride: List<Int>, dilation: List<Int>): ConvolutionOperation<RAW_TYPE>

    fun spatialConvolutionTranspose(kernel: List<Int>, padding: List<Int>, stride: List<Int>, dilation: List<Int>, outputPadding: List<Int>): ConvolutionOperation<RAW_TYPE>

    fun volumetricConvolution(kernel: List<Int>, padding: List<Int>, stride: List<Int>, dilation: List<Int>): ConvolutionOperation<RAW_TYPE>

    fun volumetricConvolutionTranspose(kernel: List<Int>, padding: List<Int>, stride: List<Int>, dilation: List<Int>, outputPadding: List<Int>): ConvolutionOperation<RAW_TYPE>

    fun toSerializableData(rawTensor: RAW_TYPE): SerializableTensorData

    fun fromCommonSerializable(commonDescriptor: CommonSerializableTensorDescriptor): Tensor

}