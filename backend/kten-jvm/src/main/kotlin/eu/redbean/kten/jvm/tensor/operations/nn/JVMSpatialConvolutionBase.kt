package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.nn.ConvolutionOperation
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import eu.redbean.kten.jvm.tensor.operations.AbstractJVMTensorOperations
import eu.redbean.kten.jvm.tensor.operations.JVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import java.util.concurrent.ConcurrentLinkedDeque

abstract class JVMSpatialConvolutionBase(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    dilationHeight: Int, dilationWidth: Int,
    ops: AbstractJVMTensorOperations
): AbstractSpatialConvLikeOpBase<JVMRawTensor, AbstractJVMTensorOperations>(
    kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth, ops
), ConvolutionOperation<JVMRawTensor> {

    protected val ones = JVMTensorOperations.createRaw(listOf(1))

    protected val columnsPool = ConcurrentLinkedDeque<JVMRawTensor>()

    protected val mapper = Im2Col2Im(
        kernelHeight, kernelWidth,
        paddingHeight, paddingWidth,
        strideHeight, strideWidth,
        dilationHeight, dilationWidth
    )

    protected abstract val columnsShape: List<Int>

    init {
        ones.mustSurviveGC = true
    }

    protected abstract fun checkWeightDimensions(weight: JVMRawTensor)
    protected abstract fun checkBiasShape(bias: JVMRawTensor?, weight: JVMRawTensor)
    protected abstract fun setInputOutputPlanes(weight: JVMRawTensor)

    protected fun shapeCheck(input: JVMRawTensor, gradOut: JVMRawTensor?, weight: JVMRawTensor, bias: JVMRawTensor?) {
        checkKernelStrideDilation()
        checkWeightDimensions(weight)
        checkBiasShape(bias, weight)
        checkInputDimensions(input)

        val dimensions = if (input.dimensions == 4) Dimensions(1) else Dimensions(0)

        inputHeight = input.shape[dimensions.height]
        inputWidth = input.shape[dimensions.width]
        setInputOutputPlanes(weight)

        calculateOutputHeightWidth()

        checkOutputSize()

        if (input.shape[dimensions.feature] != inputPlane)
            throw IllegalArgumentException("Input size at axis: ${dimensions.feature} must match weight size at axis: 0 (in transposed case) or 1, " +
                    "input shape: ${input.shape} weight shape: ${weight.shape}")

        checkGradOutShape(gradOut, dimensions)
    }

    protected fun resizeOnesIfNeeded() {
        if (ones.dimensions != 2 || ones.shape[0] * ones.shape[1] != outputHeight * outputWidth) {
            ones.inplaceResize(outputHeight, outputWidth)
            ones.inplaceFill(1f)
        }
    }

    protected fun resizeColumnsIfNeeded() {
        if (columnsPool.isEmpty()) {
            val columnsInit = JVMTensorOperations.createRaw(columnsShape)
            columnsInit.mustSurviveGC = true
            columnsPool.push(columnsInit)
        } else {
            val columns = columnsPool.pop()
            if (columns.shape != columnsShape) {
                columnsPool.onEach { it.mustSurviveGC = false }.forEach { ops.release(it as AbstractRawTensor<Any>) }
                columnsPool.clear()
                val columnsInit = JVMTensorOperations.createRaw(columnsShape)
                columnsInit.mustSurviveGC = true
                columnsPool.push(columnsInit)
            } else {
                columnsPool.push(columns)
            }
        }
    }

    protected fun getLocalColumns(): JVMRawTensor {
        var res: JVMRawTensor? = columnsPool.pollFirst()

        if (res == null) {
            res = JVMTensorOperations.createRaw(columnsShape)
            res.mustSurviveGC = true
        }

        return res
    }

    override fun cleanup() {
        ones.mustSurviveGC = false
        ops.release(ones as AbstractRawTensor<Any>)
        ops.release(*columnsPool.onEach { it.mustSurviveGC = false }.map { it as AbstractRawTensor<Any> }.toTypedArray())
        columnsPool.clear()
    }

}