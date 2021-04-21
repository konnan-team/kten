package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.nn.ConvolutionOperation
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor

abstract class OCLVolumetricConvolutionBase(
    protected val kernelDepth: Int, protected val kernelHeight: Int, protected val kernelWidth: Int,
    protected val paddingDepth: Int, protected val paddingHeight: Int, protected val paddingWidth: Int,
    protected val strideDepth: Int, protected val strideHeight: Int, protected val strideWidth: Int,
    protected val dilationDepth: Int, protected val dilationHeight: Int, protected val dilationWidth: Int,
    protected val ops: OCLTensorOperations
): ConvolutionOperation<OCLRawTensor>() {

    protected var inputPlane = -1
    protected var inputDepth = -1
    protected var inputHeight = -1
    protected var inputWidth = -1

    protected var outputPlane = -1
    protected var outputDepth = -1
    protected var outputHeight = -1
    protected var outputWidth = -1

    protected val ones = ops.createRaw(listOf(1))

    protected val mapper = OCLVol2Col2Vol(
        kernelDepth, kernelHeight, kernelWidth,
        paddingDepth, paddingHeight, paddingWidth,
        strideDepth, strideHeight, strideWidth,
        dilationDepth, dilationHeight, dilationWidth,
        ops
    )

    protected abstract val columnsShape: List<Int>

    init {
        ones.mustSurviveGC = true
    }

    protected fun checkKernelStrideDilation() {
        if (kernelDepth <= 0 || kernelHeight <= 0 || kernelWidth <= 0)
            throw IllegalArgumentException("Kernel size should be greater than zero")

        if (strideDepth <= 0 || strideHeight <= 0 || strideWidth <= 0)
            throw IllegalArgumentException("Stride should be greater than zero")

        if (dilationDepth <= 0 || dilationHeight <= 0 || dilationWidth <= 0)
            throw IllegalArgumentException("Dilation should be greater than zero")
    }

    protected fun checkWeightDimensions(weight: OCLRawTensor) {
        if (weight.dimensions != 5)
            throw IllegalArgumentException("Weight must be a 5D tensor")
    }

    protected abstract fun checkBiasShape(bias: OCLRawTensor?, weight: OCLRawTensor)

    protected fun checkInputDimensions(input: OCLRawTensor) {
        if (input.dimensions !in listOf(4, 5))
            throw IllegalArgumentException("Input must be 4D or 5D tensor")
    }

    protected abstract fun calculateOutputDepthHeightWidth()
    protected abstract fun setInputOutputPlanes(weight: OCLRawTensor)

    protected fun shapeCheck(input: OCLRawTensor, gradOut: OCLRawTensor?, weight: OCLRawTensor, bias: OCLRawTensor?) {
        checkKernelStrideDilation()
        checkWeightDimensions(weight)
        checkBiasShape(bias, weight)
        checkInputDimensions(input)

        var dimf = 0
        var dimd = 1
        var dimh = 2
        var dimw = 3

        if (input.dimensions == 5) {
            dimf++
            dimd++
            dimh++
            dimw++
        }

        inputDepth = input.shape[dimd]
        inputHeight = input.shape[dimh]
        inputWidth = input.shape[dimw]

        setInputOutputPlanes(weight)

        calculateOutputDepthHeightWidth()

        if (outputDepth < 1 || outputHeight < 1 || outputWidth < 1)
            throw IllegalArgumentException("Calculated output size: ${outputPlane} x ${outputDepth} x ${outputHeight} x ${outputWidth} " +
                    "is too small. Input size: ${inputPlane} x ${inputDepth} x ${inputHeight} x ${inputWidth}")

        if (input.shape[dimf] != inputPlane)
            throw IllegalArgumentException("Input size at axis: ${dimf} must match weight size at axis: 0 (in transposed case) or 1, " +
                    "input shape: ${input.shape} weight shape: ${weight.shape}")

        if (gradOut != null) {
            if (gradOut.shape[dimf] != outputPlane
                || gradOut.shape[dimd] != outputDepth
                || gradOut.shape[dimh] != outputHeight
                || gradOut.shape[dimw] != outputWidth)
                throw IllegalArgumentException("Invalid gradient shape, " +
                        "gradient must have shape with (F x D x H x W) = ${outputPlane} x ${outputDepth} x ${outputHeight} x ${outputWidth}, " +
                        "but got ${gradOut.shape[dimf]} x ${gradOut.shape[dimd]} x ${gradOut.shape[dimh]} x ${gradOut.shape[dimw]}")
        }
    }

    protected fun resizeOnesIfNeeded() {
        if (ones.dimensions != 3 || ones.shape[0] * ones.shape[1] * ones.shape[2] != outputDepth * outputHeight * outputWidth) {
            ones.inplaceResize(outputDepth, outputHeight, outputWidth)
            ones.inplaceFill(1f)
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun cleanup() {
        ones.mustSurviveGC = false
        ops.release(ones as AbstractRawTensor<Any>)
    }

}