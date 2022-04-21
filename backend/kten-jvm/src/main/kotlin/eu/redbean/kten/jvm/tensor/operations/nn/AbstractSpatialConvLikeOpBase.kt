package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

abstract class AbstractSpatialConvLikeOpBase<RAW_TYPE: AbstractRawTensor<*>, OPS_TYPE: TensorOperations<RAW_TYPE>>(
    protected val kernelHeight: Int, protected val kernelWidth: Int,
    protected val paddingHeight: Int, protected val paddingWidth: Int,
    protected val strideHeight: Int, protected val strideWidth: Int,
    protected val dilationHeight: Int, protected val dilationWidth: Int,
    protected val ops: OPS_TYPE
) { //TODO consider moving it to a common module (with other extractable parts)

    protected var inputPlane = -1
    protected var inputHeight = -1
    protected var inputWidth = -1

    protected var outputPlane = -1
    protected var outputHeight = -1
    protected var outputWidth = -1

    protected fun checkKernelStrideDilation() {
        if (kernelHeight <= 0 || kernelWidth <= 0)
            throw IllegalArgumentException("Kernel size should be greater than zero")

        if (strideHeight <= 0 || strideWidth <= 0)
            throw IllegalArgumentException("Stride should be greater than zero")

        if (dilationHeight <= 0 || dilationWidth <= 0)
            throw IllegalArgumentException("Dilation should be greater than zero")
    }

    protected fun checkInputDimensions(input: RAW_TYPE) {
        if (input.dimensions !in listOf(3, 4))
            throw IllegalArgumentException("Input must be 3D or 4D tensor")
    }

    protected abstract fun calculateOutputHeightWidth()

    protected fun checkOutputSize() {
        if (outputHeight < 1 || outputWidth < 1)
            throw IllegalArgumentException(
                "Calculated output size: ${outputPlane} x ${outputHeight} x ${outputWidth} is too small. " +
                        "Input size: ${inputPlane} x ${inputHeight} x ${inputWidth}"
            )
    }

    protected fun checkGradOutShape(gradOut: RAW_TYPE?, dimensions: Dimensions) {
        if (gradOut != null) {
            if (gradOut.shape[dimensions.feature] != outputPlane
                || gradOut.shape[dimensions.height] != outputHeight
                || gradOut.shape[dimensions.width] != outputWidth
            )
                throw IllegalArgumentException(
                    "Invalid gradient shape, " +
                            "gradient must have shape with (F x H x W) = ${outputPlane} x ${outputHeight} x ${outputWidth}, " +
                            "but got ${gradOut.shape[dimensions.feature]} x ${gradOut.shape[dimensions.height]} x ${gradOut.shape[dimensions.width]}"
                )
        }
    }

    data class Dimensions(
        val feature: Int, val height: Int = feature + 1, val width: Int = feature + 2
    )

}