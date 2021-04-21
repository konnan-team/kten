package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.nn.ConvolutionOperation
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import eu.redbean.kten.jvm.tensor.operations.AbstractJVMTensorOperations
import eu.redbean.kten.jvm.tensor.operations.JVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import java.util.concurrent.ConcurrentLinkedDeque

abstract class JVMSpatialConvolutionBase(
    protected val kernelHeight: Int, protected val kernelWidth: Int,
    protected val paddingHeight: Int, protected val paddingWidth: Int,
    protected val strideHeight: Int, protected val strideWidth: Int,
    protected val dilationHeight: Int, protected val dilationWidth: Int,
    protected val ops: AbstractJVMTensorOperations
): ConvolutionOperation<JVMRawTensor>() {

    protected var inputPlane = -1
    protected var inputHeight = -1
    protected var inputWidth = -1

    protected var outputPlane = -1
    protected var outputHeight = -1
    protected var outputWidth = -1

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

    protected fun checkKernelStrideDilation() {
        if (kernelHeight <= 0 || kernelWidth <= 0)
            throw IllegalArgumentException("Kernel size should be greater than zero")

        if (strideHeight <= 0 || strideWidth <= 0)
            throw IllegalArgumentException("Stride should be greater than zero")

        if (dilationHeight <= 0 || dilationWidth <= 0)
            throw IllegalArgumentException("Dilation should be greater than zero")
    }

    protected abstract fun checkWeightDimensions(weight: JVMRawTensor)
    protected abstract fun checkBiasShape(bias: JVMRawTensor?, weight: JVMRawTensor)

    protected fun checkInputDimensions(input: JVMRawTensor) {
        if (input.dimensions !in listOf(3, 4))
            throw IllegalArgumentException("Input must be 3D or 4D tensor")
    }

    protected abstract fun calculateOutputHeightWidth()
    protected abstract fun setInputOutputPlanes(weight: JVMRawTensor)

    protected fun shapeCheck(input: JVMRawTensor, gradOut: JVMRawTensor?, weight: JVMRawTensor, bias: JVMRawTensor?) {
        checkKernelStrideDilation()
        checkWeightDimensions(weight)
        checkBiasShape(bias, weight)
        checkInputDimensions(input)

        var dimf = 0
        var dimh = 1
        var dimw = 2

        if (input.dimensions == 4) {
            dimf++
            dimh++
            dimw++
        }

        inputHeight = input.shape[dimh]
        inputWidth = input.shape[dimw]
        setInputOutputPlanes(weight)

        calculateOutputHeightWidth()

        if (outputHeight < 1 || outputWidth < 1)
            throw IllegalArgumentException("Calculated output size: ${outputPlane} x ${outputHeight} x ${outputWidth} is too small. " +
                    "Input size: ${inputPlane} x ${inputHeight} x ${inputWidth}")

        if (input.shape[dimf] != inputPlane)
            throw IllegalArgumentException("Input size at axis: ${dimf} must match weight size at axis: 0 (in transposed case) or 1, " +
                    "input shape: ${input.shape} weight shape: ${weight.shape}")

        if (gradOut != null) {
            if (gradOut.shape[dimf] != outputPlane || gradOut.shape[dimh] != outputHeight || gradOut.shape[dimw] != outputWidth)
                throw IllegalArgumentException("Invalid gradient shape, " +
                        "gradient must have shape with (F x H x W) = ${outputPlane} x ${outputHeight} x ${outputWidth}, " +
                        "but got ${gradOut.shape[dimf]} x ${gradOut.shape[dimh]} x ${gradOut.shape[dimw]}")
        }
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