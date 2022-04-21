package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.nn.PoolingOperation
import eu.redbean.kten.api.tensor.operations.nn.PoolingOperation.Companion.calculateSizeOnFloats
import eu.redbean.kten.api.tensor.operations.nn.PoolingResult
import eu.redbean.kten.jvm.tensor.operations.AbstractJVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import eu.redbean.kten.jvm.tensor.store.StoreView
import java.util.stream.IntStream
import kotlin.math.ceil
import kotlin.math.floor

abstract class JVMSpatialPoolingBase(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    dilationHeight: Int, dilationWidth: Int,
    val useCeil: Boolean,
    val returnsIndices: Boolean,
    ops: AbstractJVMTensorOperations
) : AbstractSpatialConvLikeOpBase<JVMRawTensor, AbstractJVMTensorOperations>(
    kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth, ops
), PoolingOperation<JVMRawTensor> {

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

    protected fun shapeCheck(input: JVMRawTensor, gradOut: JVMRawTensor?, indices: JVMRawTensor?) {
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

    protected abstract fun updateOutputSliceView(input: StoreView, output: StoreView, indices: StoreView?)

    override fun updateOutput(input: JVMRawTensor): PoolingResult<JVMRawTensor> {
        shapeCheck(input, null, null)

        val res = if (input.dimensions == 3)
            PoolingResult(
                ops.createRaw(listOf(outputPlane, outputHeight, outputWidth)),
                if (returnsIndices) ops.createRaw(listOf(outputPlane, outputHeight, outputWidth)) else null
            )
        else
            PoolingResult(
                ops.createRaw(listOf(input.shape[0], outputPlane, outputHeight, outputWidth)),
                if (returnsIndices) ops.createRaw(listOf(input.shape[0], outputPlane, outputHeight, outputWidth)) else null
            )

        if (input.dimensions == 3) {
            updateOutputSliceView(input.asView().storeReference, res.output.asView().storeReference, res.indices?.asView()?.storeReference)
        } else {
            IntStream.range(0, input.shape[0]).parallel().forEach {
                updateOutputSliceView(input.getView(it).storeReference, res.output.getView(it).storeReference, res.indices?.getView(it)?.storeReference)
            }
        }

        return res
    }

    protected abstract fun updateGradSliceView(gradIn: StoreView, gradOut: StoreView, indices: StoreView?)

    override fun calculateGradInput(input: JVMRawTensor, gradOut: JVMRawTensor, indices: JVMRawTensor?): JVMRawTensor {
        shapeCheck(input, gradOut, indices)

        val gradIn = ops.createRaw(input.shape) //zero filled on jvm

        if (input.dimensions == 3) {
            updateGradSliceView(gradIn.asView().storeReference, gradOut.asView().storeReference, indices?.asView()?.storeReference)
        } else {
            IntStream.range(0, input.shape[0]).parallel().forEach {
                updateGradSliceView(gradIn.getView(it).storeReference, gradOut.getView(it).storeReference, indices?.getView(it)?.storeReference)
            }
        }

        return gradIn
    }

}