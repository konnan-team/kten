package eu.redbean.kten.opencl.tensor.operations

import eu.redbean.kten.api.autograd.tensor.NoGradVariable
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.autograd.utils.concatShapes
import eu.redbean.kten.api.autograd.utils.normalizeAxis
import eu.redbean.kten.api.autograd.utils.toIndexRanges
import eu.redbean.kten.api.autograd.utils.toStoreSize
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.operations.TensorOperationsGarbageCollector
import eu.redbean.kten.api.tensor.operations.nn.*
import eu.redbean.kten.api.tensor.serialization.CommonSerializableTensorDescriptor
import eu.redbean.kten.api.tensor.serialization.SerializableTensorData
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import eu.redbean.kten.opencl.tensor.operations.nn.*
import eu.redbean.kten.opencl.tensor.platform.OCLEnvironment
import eu.redbean.kten.opencl.tensor.platform.kernels.OCLKernelConstant
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.MemoryAccessOption.SOURCE
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.MemoryAccessOption.SOURCE_AND_TARGET
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor
import eu.redbean.kten.opencl.tensor.store.OCLRawTensorView
import org.jocl.blast.CLBlast
import org.jocl.blast.CLBlastLayout
import org.jocl.blast.CLBlastTranspose

class OCLTensorOperations(
    override val platformKey: String,
    environmentInitializer: () -> OCLEnvironment
): TensorOperations<OCLRawTensor> {

    internal val environment: OCLEnvironment by lazy(environmentInitializer)

    private var alreadyInGC = false

    private var gcSurvivors = listOf<OCLRawTensor>()

    override fun create(shape: List<Int>, requiresGrad: Boolean, init: (Int) -> Float): Tensor {
        return tensorFromRaw(createRaw(shape, init), requiresGrad)
    }

    override fun createFillConst(shape: List<Int>, requiresGrad: Boolean, constant: Float): Tensor {
        return tensorFromRaw(createRawFill(shape, constant), requiresGrad)
    }

    override fun createRandom(shape: List<Int>, requiresGrad: Boolean): Tensor {
        val res = createRaw(shape)
        environment.kernelStore.fillRandom(res.storeReference)
        return tensorFromRaw(res, requiresGrad)
    }

    override fun createBernoulli(shape: List<Int>, rate: Float, requiresGrad: Boolean): Tensor {
        val res = createRaw(shape)
        environment.kernelStore.fillBernoulli(res.storeReference, rate)
        return tensorFromRaw(res, requiresGrad)
    }

    @Suppress("UNCHECKED_CAST")
    private fun tensorFromRaw(rawTensor: OCLRawTensor, requiresGrad: Boolean): Tensor {
        if (requiresGrad)
            return Variable(
                this as TensorOperations<AbstractRawTensor<Any>>,
                rawTensor as AbstractRawTensor<Any>
            )
        return NoGradVariable(
            this as TensorOperations<AbstractRawTensor<Any>>,
            rawTensor as AbstractRawTensor<Any>
        )
    }

    override fun garbageCollector(): TensorOperationsGarbageCollector {
        if (alreadyInGC) //overlapping case
            return TensorOperationsGarbageCollector {  }

        val instances = gcSurvivors.toMutableList()
        environment.instanceCollector = { instances.add(it) }
        alreadyInGC = true
        return TensorOperationsGarbageCollector {
            environment.instanceCollector = {}
            alreadyInGC = false
            release(instances)
            gcSurvivors = instances.filter { it.referenced() }
            instances.clear()
        }
    }

    override fun markSurviveGC(rawTensor: AbstractRawTensor<Any>) {
        (rawTensor as OCLRawTensor).mustSurviveGC = true
    }

    override fun markReleasableInGC(rawTensor: AbstractRawTensor<Any>) {
        (rawTensor as OCLRawTensor).mustSurviveGC = false
    }

    override fun createRaw(shape: List<Int>, init: (Int) -> Float): OCLRawTensor {
        return OCLRawTensor(shape, environment.memoryObjectOf(FloatArray(shape.toStoreSize(), init)), environment)
    }

    override fun createRaw(shape: List<Int>): OCLRawTensor {
        return OCLRawTensor(shape, environment.memoryObject(shape.toStoreSize()), environment)
    }

    override fun createRawFill(shape: List<Int>, constant: Float): OCLRawTensor {
        val res = createRaw(shape)
        res.storeReference.fill(constant)
        return res
    }

    override fun release(vararg rawTensors: AbstractRawTensor<Any>) {
        release(rawTensors.map { it as OCLRawTensor })
    }

    internal fun release(rawTensors: Iterable<OCLRawTensor>) {
        rawTensors.forEach { it.release() }
    }

    override fun zerosLike(rawTensor: OCLRawTensor): OCLRawTensor {
        val res = environment.memoryObject(rawTensor.storeReference.size)
        res.fill(0f)
        return OCLRawTensor(rawTensor.shape.toList(), res, environment)
    }

    override fun zeroOut(rawTensor: OCLRawTensor) {
        rawTensor.storeReference.fill(0f)
    }

    override fun incrementRef(rawTensor: OCLRawTensor) {
        rawTensor.incrementRef()
    }

    override fun pow(constant: Float, rawTensor: OCLRawTensor): OCLRawTensor {
        val res = environment.memoryObject(rawTensor.storeReference.size)
        environment.kernelStore.constTensorOp(constant, rawTensor.storeReference, res, OCLKernelConstant.POW)
        return OCLRawTensor(rawTensor.shape.toList(), res, environment)
    }

    override fun concat(axis: Int, inputs: List<OCLRawTensor>): OCLRawTensor {
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

    override fun gemm(
        addMatrix: OCLRawTensor,
        matrix1: OCLRawTensor,
        matrix2: OCLRawTensor,
        alpha: Float,
        beta: Float,
        transposeFirst: Boolean,
        transposeSecond: Boolean
    ): OCLRawTensor {
        gemmViews(addMatrix.asView(), matrix1.asView(), matrix2.asView(), alpha, beta, transposeFirst, transposeSecond)
        return addMatrix
    }

    fun gemmViews(
        addMatrix: OCLRawTensorView,
        matrix1: OCLRawTensorView,
        matrix2: OCLRawTensorView,
        alpha: Float,
        beta: Float,
        transposeFirst: Boolean,
        transposeSecond: Boolean
    ): OCLRawTensorView {
        val (m, k) = matrix1.shape
        val n = matrix2.shape[1]
        val mTransAware = if (transposeFirst) matrix1.shape[1] else matrix1.shape[0]
        val kTransAware = if (transposeFirst) matrix1.shape[0] else matrix1.shape[1]
        val nTransAware = if (transposeSecond) matrix2.shape[0] else matrix2.shape[1]
        CLBlast.CLBlastSgemm(
            CLBlastLayout.CLBlastLayoutRowMajor,
            if (transposeFirst) CLBlastTranspose.CLBlastTransposeYes else CLBlastTranspose.CLBlastTransposeNo,
            if (transposeSecond) CLBlastTranspose.CLBlastTransposeYes else CLBlastTranspose.CLBlastTransposeNo,
            mTransAware.toLong(), nTransAware.toLong(), kTransAware.toLong(),
            alpha,
            matrix1.storeReference.getMemoryObject(SOURCE), matrix1.storeReference.offset, k.toLong(),
            matrix2.storeReference.getMemoryObject(SOURCE), matrix2.storeReference.offset, n.toLong(),
            beta,
            addMatrix.storeReference.getMemoryObject(SOURCE_AND_TARGET), addMatrix.storeReference.offset, nTransAware.toLong(),
            environment.commandQueue,
            null
        )
        return addMatrix
    }

    override fun gemv(addVector: OCLRawTensor, matrix: OCLRawTensor, vector: OCLRawTensor, alpha: Float, beta: Float, transposeMatrix: Boolean): OCLRawTensor {
        gemvViews(addVector.asView(), matrix.asView(), vector.asView(), alpha, beta, transposeMatrix)
        return addVector
    }

    fun gemvViews(addVector: OCLRawTensorView,
                  matrix: OCLRawTensorView,
                  vector: OCLRawTensorView,
                  alpha: Float, beta: Float,
                  transposeMatrix: Boolean): OCLRawTensorView {
        val (m, n) = matrix.shape
        CLBlast.CLBlastSgemv(
            CLBlastLayout.CLBlastLayoutRowMajor,
            if (transposeMatrix) CLBlastTranspose.CLBlastTransposeYes else CLBlastTranspose.CLBlastTransposeNo,
            m.toLong(), n.toLong(),
            alpha,
            matrix.storeReference.getMemoryObject(SOURCE), matrix.storeReference.offset, n.toLong(),
            vector.storeReference.getMemoryObject(SOURCE), vector.storeReference.offset, 1L,
            beta,
            addVector.storeReference.getMemoryObject(SOURCE_AND_TARGET), addVector.storeReference.offset, 1L,
            environment.commandQueue,
            null
        )
        return addVector
    }

    override fun gemmBatched(
        addTensor: OCLRawTensor,
        tensor1: OCLRawTensor,
        tensor2: OCLRawTensor,
        alpha: Float,
        beta: Float,
        transposeFirst: Boolean,
        transposeSecond: Boolean
    ): OCLRawTensor {
        val (batch, m, k) = tensor1.shape
        val n = tensor2.shape[2]
        val mTransAware = if (transposeFirst) tensor1.shape[2] else tensor1.shape[1]
        val kTransAware = if (transposeFirst) tensor1.shape[1] else tensor1.shape[2]
        val nTransAware = if (transposeSecond) tensor2.shape[1] else tensor2.shape[2]
        val t1Offset = tensor1.shape[1] * tensor1.shape[2]
        val t2Offset = tensor2.shape[1] * tensor2.shape[2]
        val addTOffset = addTensor.shape[1] * addTensor.shape[2]
        CLBlast.CLBlastSgemmBatched(
            CLBlastLayout.CLBlastLayoutRowMajor,
            if (transposeFirst) CLBlastTranspose.CLBlastTransposeYes else CLBlastTranspose.CLBlastTransposeNo,
            if (transposeSecond) CLBlastTranspose.CLBlastTransposeYes else CLBlastTranspose.CLBlastTransposeNo,
            mTransAware.toLong(), nTransAware.toLong(), kTransAware.toLong(),
            FloatArray(batch) { alpha },
            tensor1.storeReference.getMemoryObject(SOURCE), LongArray(batch) { (it * t1Offset).toLong() }, k.toLong(),
            tensor2.storeReference.getMemoryObject(SOURCE), LongArray(batch) { (it * t2Offset).toLong() }, n.toLong(),
            FloatArray(batch) { beta },
            addTensor.storeReference.getMemoryObject(SOURCE_AND_TARGET), LongArray(batch) { (it * addTOffset).toLong() }, nTransAware.toLong(),
            batch.toLong(),
            environment.commandQueue,
            null
        )
        return addTensor
    }

    override fun ger(vector1: OCLRawTensor, vector2: OCLRawTensor, matrix: OCLRawTensor, alpha: Float): OCLRawTensor {
        val (m, n) = matrix.shape
        CLBlast.CLBlastSger(
            CLBlastLayout.CLBlastLayoutRowMajor,
            m.toLong(), n.toLong(),
            alpha,
            vector1.storeReference.getMemoryObject(SOURCE), 0L, 1L,
            vector2.storeReference.getMemoryObject(SOURCE), 0L, 1L,
            matrix.storeReference.getMemoryObject(SOURCE_AND_TARGET), 0L, n.toLong(),
            environment.commandQueue,
            null
        )
        return matrix
    }

    override fun spatialConvolution(kernel: List<Int>, padding: List<Int>, stride: List<Int>, dilation: List<Int>): ConvolutionOperation<OCLRawTensor> {
        return OCLSpatialConvolution(
            kernel[0], kernel[1],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            this
        )
    }

    override fun spatialConvolutionTranspose(
        kernel: List<Int>,
        padding: List<Int>,
        stride: List<Int>,
        dilation: List<Int>,
        outputPadding: List<Int>
    ): ConvolutionOperation<OCLRawTensor> {
        return OCLSpatialConvolutionTranspose(
            kernel[0], kernel[1],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            outputPadding[0], outputPadding[1],
            this
        )
    }

    override fun spatialPooling(
        kernel: List<Int>,
        padding: List<Int>,
        stride: List<Int>,
        dilation: List<Int>,
        options: PoolingOptions
    ): PoolingOperation<OCLRawTensor> {
        return when (options.type) {
            PoolingType.MAX -> OCLSpatialMaxPooling(
                kernel[0], kernel[1],
                padding[0], padding[1],
                stride[0], stride[1],
                dilation[0], dilation[1],
                options.useCeil,
                this
            )
            PoolingType.AVG -> OCLSpatialAvgPooling(
                kernel[0], kernel[1],
                padding[0], padding[1],
                stride[0], stride[1],
                options.useCeil,
                options.includePadding,
                this
            )
        }
    }

    override fun volumetricConvolution(kernel: List<Int>, padding: List<Int>, stride: List<Int>, dilation: List<Int>): ConvolutionOperation<OCLRawTensor> {
        return OCLVolumetricConvolution(
            kernel[0], kernel[1], kernel[2],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
            this
        )
    }

    override fun volumetricConvolutionTranspose(
        kernel: List<Int>,
        padding: List<Int>,
        stride: List<Int>,
        dilation: List<Int>,
        outputPadding: List<Int>
    ): ConvolutionOperation<OCLRawTensor> {
        return OCLVolumetricConvolutionTranspose(
            kernel[0], kernel[1], kernel[2],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
            outputPadding[0], outputPadding[1], outputPadding[2],
            this
        )
    }

    override fun batchNorm(axis: Int, momentum: Float, epsilon: Float, training: Boolean): BatchNormOperation<OCLRawTensor> {
        return OCLBatchNorm(axis, momentum, epsilon, training, this)
    }

    override fun upsample(upsampleType: UpsampleType, scale: Int): Upsample2DOperation<OCLRawTensor> {
        return when (upsampleType) {
            UpsampleType.NEAREST -> OCLUpsample2DNearest(scale, this)
        }
    }

    override fun toSerializableData(rawTensor: OCLRawTensor): SerializableTensorData {
        rawTensor.storeReference.readToArray()
        return SerializableTensorData(rawTensor.shape.toList(), rawTensor.storeReference.jvmArray.copyOf())
    }

    @Suppress("UNCHECKED_CAST")
    override fun fromCommonSerializable(commonDescriptor: CommonSerializableTensorDescriptor): Tensor {
        val dataMemObject = environment.memoryObjectOf(commonDescriptor.data.data.copyOf())
        val dataRaw = OCLRawTensor(commonDescriptor.data.shape.toList(), dataMemObject, environment)
        val gradData = commonDescriptor.gradientData
        if (gradData != null) {
            val gradMemObject = environment.memoryObjectOf(gradData.data)
            val gradRaw = OCLRawTensor(gradData.shape.toList(), gradMemObject, environment)
            return Variable(this as TensorOperations<AbstractRawTensor<Any>>, dataRaw as AbstractRawTensor<Any>, gradRaw as AbstractRawTensor<Any>)
        }
        return NoGradVariable(this as TensorOperations<AbstractRawTensor<Any>>, dataRaw as AbstractRawTensor<Any>)
    }
}