package eu.redbean.kten.jvm.tensor.operations

import eu.redbean.kten.api.autograd.tensor.NoGradVariable
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.autograd.utils.concatShapes
import eu.redbean.kten.api.autograd.utils.normalizeAxis
import eu.redbean.kten.api.autograd.utils.toIndexRanges
import eu.redbean.kten.api.autograd.utils.toStoreSize
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.operations.TensorOperationsGarbageCollector
import eu.redbean.kten.api.tensor.operations.nn.ConvolutionOperation
import eu.redbean.kten.api.tensor.serialization.CommonSerializableTensorDescriptor
import eu.redbean.kten.api.tensor.serialization.SerializableTensorData
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import eu.redbean.kten.jvm.tensor.operations.nn.JVMSpatialConvolution
import eu.redbean.kten.jvm.tensor.operations.nn.JVMSpatialConvolutionTranspose
import eu.redbean.kten.jvm.tensor.operations.nn.JVMVolumetricConvolution
import eu.redbean.kten.jvm.tensor.operations.nn.JVMVolumetricConvolutionTranspose
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import eu.redbean.kten.jvm.tensor.store.JVMRawTensorView
import java.util.stream.IntStream
import kotlin.math.pow
import kotlin.random.Random
import kotlin.random.asJavaRandom

abstract class AbstractJVMTensorOperations: TensorOperations<JVMRawTensor> {

    @Suppress("UNCHECKED_CAST")
    override fun create(shape: List<Int>, requiresGrad: Boolean, init: (Int) -> Float): Tensor {
        if (requiresGrad) {
            return Variable(
                this as TensorOperations<AbstractRawTensor<Any>>,
                JVMRawTensor(shape, FloatArray(shape.toStoreSize(), init), platformKey) as AbstractRawTensor<Any>
            )
        }
        return NoGradVariable(
            this as TensorOperations<AbstractRawTensor<Any>>,
            JVMRawTensor(shape, FloatArray(shape.toStoreSize(), init), platformKey) as AbstractRawTensor<Any>
        )
    }

    @Suppress("UNCHECKED_CAST")
    override fun createRandom(shape: List<Int>, requiresGrad: Boolean): Tensor {
        val store = FloatArray(shape.toStoreSize()) {
            Random.asJavaRandom().nextGaussian().toFloat()
        }
        if (requiresGrad) {
            return Variable(
                this as TensorOperations<AbstractRawTensor<Any>>,
                JVMRawTensor(shape, store, platformKey) as AbstractRawTensor<Any>
            )
        }
        return NoGradVariable(
            this as TensorOperations<AbstractRawTensor<Any>>,
            JVMRawTensor(shape, store, platformKey) as AbstractRawTensor<Any>
        )
    }

    @Suppress("UNCHECKED_CAST")
    override fun createBernoulli(shape: List<Int>, rate: Float, requiresGrad: Boolean): Tensor {
        val store = FloatArray(shape.toStoreSize()) {
            if (Random.nextFloat() < rate) 1.0f else 0.0f
        }
        if (requiresGrad)
            return Variable(
                this as TensorOperations<AbstractRawTensor<Any>>,
                JVMRawTensor(shape, store, platformKey) as AbstractRawTensor<Any>
            )
        return NoGradVariable(
            this as TensorOperations<AbstractRawTensor<Any>>,
            JVMRawTensor(shape, store, platformKey) as AbstractRawTensor<Any>
        )
    }

    override fun release(vararg rawTensors: AbstractRawTensor<Any>) {
        //no-op on jvm
    }

    override fun garbageCollector(): TensorOperationsGarbageCollector {
        return TensorOperationsGarbageCollector {}
    }

    override fun zerosLike(rawTensor: JVMRawTensor): JVMRawTensor {
        return JVMRawTensor(rawTensor.shape, FloatArray(rawTensor.storeReference.size) { 0.0f }, platformKey)
    }

    override fun zeroOut(rawTensor: JVMRawTensor) {
        IntStream.range(0, rawTensor.storeReference.size).parallel()
            .forEach { rawTensor.storeReference[it] = 0.0f }
    }

    override fun incrementRef(rawTensor: JVMRawTensor) {
        // no-op for jvm
    }

    override fun pow(constant: Float, rawTensor: JVMRawTensor): JVMRawTensor {
        return JVMRawTensor(rawTensor.shape, FloatArray(rawTensor.storeReference.size) { constant.pow(rawTensor.storeReference[it]) }, platformKey)
    }

    override fun createRaw(shape: List<Int>, init: (Int) -> Float): JVMRawTensor {
        return JVMRawTensor(shape, FloatArray(shape.toStoreSize(), init), platformKey)
    }

    override fun createRaw(shape: List<Int>): JVMRawTensor {
        return JVMRawTensor(shape, FloatArray(shape.toStoreSize()), platformKey)
    }

    override fun concat(axis: Int, inputs: List<JVMRawTensor>): JVMRawTensor {
        val shapes = inputs.map { it.shape }
        val resShape = concatShapes(shapes, axis)
        val normAxis = inputs[0].shape.normalizeAxis(axis)
        val targetRanges = resShape.toIndexRanges()
        val res = createRaw(resShape)

        var last = 0
        for (i in shapes.indices) {
            targetRanges[normAxis] = last until (last + shapes[i][normAxis])
            res[targetRanges] = inputs[i]
            last += shapes[i][normAxis]
        }
        return res
    }

    private class BatchedMatrixView(private val tensor: JVMRawTensorView) {
        private val cols = tensor.shape[2]
        private val rows = tensor.shape[1]

        val size
            get() = tensor.storeReference.size

        operator fun get(b: Int, i: Int, j: Int): Float = tensor.storeReference[(b * rows + i) * cols + j]
        operator fun set(b: Int, i: Int, j: Int, value: Float) {
            tensor.storeReference[(b * rows + i) * cols + j] = value
        }
        operator fun get(i: Int): Float = tensor.storeReference[i]
        operator fun set(i: Int, value: Float) {
            tensor.storeReference[i] = value
        }
    }

    override fun gemm(
        addMatrix: JVMRawTensor,
        matrix1: JVMRawTensor,
        matrix2: JVMRawTensor,
        alpha: Float,
        beta: Float,
        transposeFirst: Boolean,
        transposeSecond: Boolean
    ): JVMRawTensor {
        gemmViews(addMatrix.asView(), matrix1.asView(), matrix2.asView(), alpha, beta, transposeFirst, transposeSecond)
        return addMatrix
    }

    fun gemmViews(
        addMatrix: JVMRawTensorView,
        matrix1: JVMRawTensorView,
        matrix2: JVMRawTensorView,
        alpha: Float,
        beta: Float,
        transposeFirst: Boolean,
        transposeSecond: Boolean
    ) {
        gemmBatchedViews(
            addMatrix.unsqueeze(0),
            matrix1.unsqueeze(0),
            matrix2.unsqueeze(0),
            alpha, beta,
            transposeFirst,
            transposeSecond
        )
    }

    override fun gemv(addVector: JVMRawTensor, matrix: JVMRawTensor, vector: JVMRawTensor, alpha: Float, beta: Float, transposeMatrix: Boolean): JVMRawTensor {
        gemvViews(addVector.asView(), matrix.asView(), vector.asView(), alpha, beta, transposeMatrix)
        return addVector
    }

    fun gemvViews(
        addVector: JVMRawTensorView,
        matrix: JVMRawTensorView,
        vector: JVMRawTensorView,
        alpha: Float,
        beta: Float,
        transposeMatrix: Boolean
    ) {
        if (beta != 1.0f)
            addVector *= beta

        if (alpha == 0.0f)
            return

        val (m, n) = matrix.shape

        val mA = BatchedMatrixView(matrix.unsqueeze(0))
        val y = addVector.storeReference
        val x = vector.storeReference

        if (transposeMatrix.not()) {
            IntStream.range(0, n)
                .forEach { j ->
                    if (x[j] != 0.0f) {
                        val temp = alpha * x[j]
                        for (i in 0 until m)
                            y[i] += temp * mA[0, i, j]
                    }
                }
        } else {
            IntStream.range(0, n)
                .forEach { j ->
                    var temp = 0.0f
                    for (i in 0 until m)
                        temp += mA[0, i, j] * x[i]
                    y[j] += alpha * temp
                }
        }
    }

    override fun gemmBatched(
        addTensor: JVMRawTensor,
        tensor1: JVMRawTensor,
        tensor2: JVMRawTensor,
        alpha: Float,
        beta: Float,
        transposeFirst: Boolean,
        transposeSecond: Boolean
    ): JVMRawTensor {
        gemmBatchedViews(addTensor.asView(), tensor1.asView(), tensor2.asView(), alpha, beta, transposeFirst, transposeSecond)
        return addTensor
    }

    fun gemmBatchedViews(
        addTensor: JVMRawTensorView,
        tensor1: JVMRawTensorView,
        tensor2: JVMRawTensorView,
        alpha: Float,
        beta: Float,
        transposeFirst: Boolean,
        transposeSecond: Boolean
    ) {
        if (alpha == 0.0f) {
            if (beta != 1.0f)
                addTensor *= beta
            return
        }

        val batchSize = addTensor.shape[0]

        val m = if (transposeFirst) tensor1.shape[2] else tensor1.shape[1]
        val n = if (transposeSecond) tensor2.shape[1] else tensor2.shape[2]
        val k = if (transposeFirst) tensor1.shape[1] else tensor1.shape[2]

        val mA = BatchedMatrixView(tensor1)
        val mB = BatchedMatrixView(tensor2)
        val mC = BatchedMatrixView(addTensor)

        if (transposeFirst.not()) {
            if (beta == 0.0f)
                IntStream.range(0, mC.size).parallel().forEach { mC[it] = 0.0f }
            else if (beta != 1.0f)
                IntStream.range(0, mC.size).parallel().forEach { mC[it] *= beta }

            IntStream.range(0, batchSize).parallel()
                .forEach { batch ->
                    IntStream.range(0, n).parallel()
                        .forEach { j ->
                            for (l in 0 until k) {
                                val temp = alpha * if (transposeSecond) mB[batch, j, l] else mB[batch, l, j]
                                for (i in 0 until m) {
                                    mC[batch, i, j] += temp * mA[batch, i, l]
                                }
                            }
                        }
                }

        } else {
            IntStream.range(0, batchSize).parallel()
                .forEach { batch ->
                    IntStream.range(0, n).parallel()
                        .forEach { j ->
                            for (i in 0 until m) {
                                var temp = 0.0f
                                for (l in 0 until k) {
                                    temp += mA[batch, l, i] * if (transposeSecond) mB[batch, j, l] else mB[batch, l, j]
                                }

                                if (beta == 0.0f)
                                    mC[batch, i, j] = alpha * temp
                                else
                                    mC[batch, i, j] = alpha * temp + beta * mC[batch, i, j]
                            }
                        }
                }

        }
    }

    override fun ger(vector1: JVMRawTensor, vector2: JVMRawTensor, matrix: JVMRawTensor, alpha: Float): JVMRawTensor {
        val (m, n) = matrix.shape
        val mA = BatchedMatrixView(matrix.view(1, m, n))
        val x = vector1.storeReference
        val y = vector2.storeReference

        IntStream.range(0, n).parallel()
            .forEach { j ->
                if (y[j] != 0.0f) {
                    val temp = alpha * y[j]
                    for (i in 0 until m)
                        mA[0, i, j] += x[i] * temp
                }
            }

        return matrix
    }

    override fun spatialConvolution(kernel: List<Int>, padding: List<Int>, stride: List<Int>, dilation: List<Int>): ConvolutionOperation<JVMRawTensor> {
        return JVMSpatialConvolution(
            kernel[0], kernel[1],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            this
        )
    }

    override fun spatialConvolutionTranspose(kernel: List<Int>, padding: List<Int>, stride: List<Int>, dilation: List<Int>, outputPadding: List<Int>): ConvolutionOperation<JVMRawTensor> {
        return JVMSpatialConvolutionTranspose(
            kernel[0], kernel[1],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            outputPadding[0], outputPadding[1],
            this
        )
    }

    override fun volumetricConvolution(kernel: List<Int>, padding: List<Int>, stride: List<Int>, dilation: List<Int>): ConvolutionOperation<JVMRawTensor> {
        return JVMVolumetricConvolution(
            kernel[0], kernel[1], kernel[2],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
            this
        )
    }

    override fun volumetricConvolutionTranspose(kernel: List<Int>, padding: List<Int>, stride: List<Int>, dilation: List<Int>, outputPadding: List<Int>): ConvolutionOperation<JVMRawTensor> {
        return JVMVolumetricConvolutionTranspose(
            kernel[0], kernel[1], kernel[2],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
            outputPadding[0], outputPadding[1], outputPadding[2],
            this
        )
    }

    override fun toSerializableData(rawTensor: JVMRawTensor): SerializableTensorData {
        return SerializableTensorData(rawTensor.shape.toList(), rawTensor.storeReference.copyOf())
    }

    @Suppress("UNCHECKED_CAST")
    override fun fromCommonSerializable(commonDescriptor: CommonSerializableTensorDescriptor): Tensor {
        val data = JVMRawTensor(commonDescriptor.data.shape.toList(), commonDescriptor.data.data.copyOf(), platformKey)
        val gradData = commonDescriptor.gradientData
        if (gradData != null) {
            val grad = JVMRawTensor(gradData.shape.toList(), gradData.data.copyOf(), platformKey)
            return Variable(this as TensorOperations<AbstractRawTensor<Any>>, data as AbstractRawTensor<Any>, grad as AbstractRawTensor<Any>)
        }
        return NoGradVariable(this as TensorOperations<AbstractRawTensor<Any>>, data as AbstractRawTensor<Any>)
    }

}
