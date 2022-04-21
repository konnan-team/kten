package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.nn.ConvolutionOperation
import eu.redbean.kten.jvm.tensor.operations.nn.AbstractSpatialConvLikeOpBase
import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor

abstract class OCLSpatialConvolutionBase(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    dilationHeight: Int, dilationWidth: Int,
    ops: OCLTensorOperations
): AbstractSpatialConvLikeOpBase<OCLRawTensor, OCLTensorOperations>(
    kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth, ops
), ConvolutionOperation<OCLRawTensor> {

    protected abstract val columnsShape: List<Int>

    protected val ones = ops.createRaw(listOf(1))

    protected val mapper = OCLIm2Col2Im(
        kernelHeight, kernelWidth,
        paddingHeight, paddingWidth,
        strideHeight, strideWidth,
        dilationHeight, dilationWidth,
        ops
    )

    init {
        ones.mustSurviveGC = true
    }

    protected abstract fun checkWeightDimensions(weight: OCLRawTensor)
    protected abstract fun checkBiasShape(bias: OCLRawTensor?, weight: OCLRawTensor)
    protected abstract fun setInputOutputPlanes(weight: OCLRawTensor)

    protected fun shapeCheck(input: OCLRawTensor, gradOut: OCLRawTensor?, weight: OCLRawTensor, bias: OCLRawTensor?) {
        checkKernelStrideDilation()
        checkWeightDimensions(weight)
        checkBiasShape(bias, weight)
        checkInputDimensions(input)

        val dimensions = if (input.dimensions == 4) Dimensions(1) else Dimensions(0)

        inputHeight = input.shape[dimensions.height]
        inputWidth = input.shape[dimensions.width]
        setInputOutputPlanes(weight)

        calculateOutputHeightWidth()

        if (outputHeight < 1 || outputWidth < 1)
            throw IllegalArgumentException("Calculated output size: ${outputPlane} x ${outputHeight} x ${outputWidth} is too small. " +
                    "Input size: ${inputPlane} x ${inputHeight} x ${inputWidth}")

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

    override fun cleanup() {
        ones.mustSurviveGC = false
        ones.release()
    }
}