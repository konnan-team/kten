package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.nn.PoolingOperation
import eu.redbean.kten.api.tensor.operations.nn.PoolingOperation.Companion.calculateSizeOnFloats
import eu.redbean.kten.jvm.tensor.operations.nn.AbstractSpatialConvLikeOpBase
import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor
import kotlin.math.ceil
import kotlin.math.floor

abstract class OCLSpatialPoolingBase(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    dilationHeight: Int, dilationWidth: Int,
    val useCeil: Boolean,
    ops: OCLTensorOperations
): AbstractSpatialConvLikeOpBase<OCLRawTensor, OCLTensorOperations>(
    kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth, ops
), PoolingOperation<OCLRawTensor> {

    override fun calculateOutputHeightWidth() {
        val heightFloat = calculateSizeOnFloats(
            inputHeight.toFloat(),
            dilationHeight.toFloat(),
            kernelHeight.toFloat(),
            paddingHeight.toFloat(),
            strideHeight.toFloat()
        )
        val widthFloat = calculateSizeOnFloats(
            inputWidth.toFloat(),
            dilationWidth.toFloat(),
            kernelWidth.toFloat(),
            paddingWidth.toFloat(),
            strideWidth.toFloat()
        )
        if (useCeil) {
            outputHeight = ceil(heightFloat).toInt()
            outputWidth = ceil(widthFloat).toInt()
        } else {
            outputHeight = floor(heightFloat).toInt()
            outputWidth = floor(widthFloat).toInt()
        }
    }

    protected fun shapeCheck(input: OCLRawTensor, gradOut: OCLRawTensor?, indices: OCLRawTensor?) {
        checkKernelStrideDilation()
        checkInputDimensions(input)

        if (paddingHeight > kernelHeight / 2 || paddingWidth > kernelWidth / 2) {
            throw IllegalArgumentException(
                "Padding should be smaller than kernel size / 2, " +
                        "but got kernel: ($kernelHeight, $kernelWidth) and padding: ($paddingHeight, $paddingWidth)"
            )
        }

        val dimensions = if (input.dimensions == 4) Dimensions(1) else Dimensions(0)

        inputHeight = input.shape[dimensions.height]
        inputWidth = input.shape[dimensions.width]
        inputPlane = input.shape[dimensions.feature]
        outputPlane = inputPlane

        calculateOutputHeightWidth()

        if (paddingHeight > 0 || paddingWidth > 0) {
            if ((outputHeight - 1) * strideHeight >= inputHeight + paddingHeight)
                outputHeight--
            if ((outputWidth - 1) * strideWidth >= inputWidth + paddingWidth)
                outputWidth--
        }

        checkOutputSize()

        checkGradOutShape(gradOut, dimensions)

        if (indices != null) {
            if (indices.shape[dimensions.feature] != outputPlane
                || indices.shape[dimensions.height] != outputHeight
                || indices.shape[dimensions.width] != outputWidth
            ) {
                throw IllegalArgumentException(
                    "Invalid indices shape, " +
                            "indices must have shape width (F x H x W) = $outputPlane x $outputHeight x $outputWidth, " +
                            "but got ${indices.shape[dimensions.feature]} x ${indices.shape[dimensions.height]} x ${indices.shape[dimensions.width]}"
                )
            }
        }
    }


}