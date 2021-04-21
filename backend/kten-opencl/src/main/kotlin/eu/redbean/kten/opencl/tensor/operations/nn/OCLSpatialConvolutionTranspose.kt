package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.api.autograd.utils.squeeze
import eu.redbean.kten.api.autograd.utils.unsqueeze
import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor

class OCLSpatialConvolutionTranspose(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    dilationHeight: Int, dilationWidth: Int,
    private val outputPaddingHeight: Int, private val outputPaddingWidth: Int,
    ops: OCLTensorOperations
): OCLSpatialConvolutionBase(
    kernelHeight, kernelWidth,
    paddingHeight, paddingWidth,
    strideHeight, strideWidth,
    dilationHeight, dilationWidth,
    ops
) {
    override val columnsShape: List<Int>
        get() = listOf(outputPlane * kernelHeight * kernelWidth, inputHeight * inputWidth)

    override fun checkWeightDimensions(weight: OCLRawTensor) {
        if (weight.dimensions !in listOf(2, 4))
            throw IllegalArgumentException("Weight must be 2D or 4D tensor")
    }

    override fun checkBiasShape(bias: OCLRawTensor?, weight: OCLRawTensor) {
        if (bias != null && (bias.dimensions != 1 || bias.shape[0] != weight.shape[1]))
            throw IllegalArgumentException(
                "Bias must have shape: [weight.shape[1]], " +
                        "but got bias with shape: ${bias.shape} and weight with shape: ${weight.shape}"
            )
    }

    override fun calculateOutputHeightWidth() {
        if ((outputPaddingWidth >= strideWidth && outputPaddingWidth >= dilationWidth)
            || (outputPaddingHeight >= strideHeight && outputPaddingHeight >= dilationHeight))
            throw IllegalArgumentException("Output padding must be smaller than either stride or dilation")

        outputHeight = (inputHeight - 1) * strideHeight - 2 * paddingHeight + (dilationHeight * (kernelHeight - 1) + 1) + outputPaddingHeight
        outputWidth = (inputWidth - 1) * strideWidth - 2 * paddingWidth + (dilationWidth * (kernelWidth - 1) + 1) + outputPaddingWidth
    }

    override fun setInputOutputPlanes(weight: OCLRawTensor) {
        inputPlane = weight.shape[0]
        outputPlane = weight.shape[1]
    }

    override fun calculateOutput(input: OCLRawTensor, weight: OCLRawTensor, bias: OCLRawTensor?): OCLRawTensor {
        shapeCheck(input, null, weight, bias)

        var batched = true

        if (input.dimensions == 3) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
        }

        val batchSize = input.shape[0]

        val output = ops.createRaw(listOf(batchSize, outputPlane, outputHeight, outputWidth))

        val columns = ops.createRaw(columnsShape)

        resizeOnesIfNeeded()

        for (b in 0 until batchSize) {
            val inputView = input.getView(b)
            val outputView = output.getView(b)

            ops.gemmViews(
                columns.asView(),
                weight.view(weight.shape[0], -1),
                inputView.view(-1, columns.shape[1]),
                1f, 0f, true, false
            )

            mapper.col2im(columns.storeReference, outputPlane, outputHeight, outputWidth, inputHeight, inputWidth, outputView.storeReference)

            if (bias != null) {
                ops.gemmViews(
                    outputView.view(-1, outputHeight*outputWidth),
                    bias.view(bias.shape[0], 1),
                    ones.view(1, outputHeight*outputWidth),
                    1f, 1f, false, false
                )
            }
        }

        if (!batched) {
            input.shape = input.shape.squeeze(0)
            output.shape = output.shape.squeeze(0)
        }

        return output
    }

    override fun calculateGradInput(input: OCLRawTensor, gradOut: OCLRawTensor, weight: OCLRawTensor): OCLRawTensor {
        shapeCheck(input, gradOut, weight, null)

        var batched = true
        if (input.dimensions == 3) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
            gradOut.shape = gradOut.shape.unsqueeze(0)
        }

        val batchSize = input.shape[0]

        val gradInput = ops.createRaw(listOf(batchSize, inputPlane, inputHeight, inputWidth))

        val gradColumns = ops.createRaw(columnsShape)

        for (b in 0 until batchSize) {
            val gradInputView = gradInput.getView(b)
            val gradOutputView = gradOut.getView(b)

            mapper.im2col(gradOutputView.storeReference, outputPlane, outputHeight, outputWidth, gradColumns.storeReference)

            ops.gemmViews(
                gradInputView.view(-1, gradColumns.shape[1]),
                weight.view(weight.shape[0], -1),
                gradColumns.asView(),
                1f, 0f, false, false
            )
        }

        if (!batched) {
            gradOut.shape = gradOut.shape.squeeze(0)
            input.shape = input.shape.squeeze(0)
            gradInput.shape = gradInput.shape.squeeze(0)
        }

        return gradInput
    }

    override fun accumulateGradParams(input: OCLRawTensor, gradOut: OCLRawTensor, gradWeight: OCLRawTensor, gradBias: OCLRawTensor?, scale: Float) {
        shapeCheck(input, gradOut, gradWeight, gradBias)

        var batched = true
        if (input.dimensions == 3) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
            gradOut.shape = gradOut.shape.unsqueeze(0)
        }

        resizeOnesIfNeeded()

        val batchSize = input.shape[0]
        val columns = ops.createRaw(columnsShape)

        for (b in 0 until batchSize) {
            val inputView = input.getView(b)
            val gradOutputView = gradOut.getView(b)

            mapper.im2col(gradOutputView.storeReference, outputPlane, outputHeight, outputWidth, columns.storeReference)

            ops.gemmViews(
                gradWeight.view(gradWeight.shape[0], -1),
                inputView.view(-1, columns.shape[1]),
                columns.asView(),
                scale, 1f, false, true
            )

            if (gradBias != null) {
                ops.gemvViews(
                    gradBias.asView(),
                    gradOutputView.view(gradBias.shape[0], -1),
                    ones.view(outputHeight*outputWidth),
                    scale, 1f, false
                )
            }
        }

        if (!batched) {
            input.shape = input.shape.squeeze(0)
            gradOut.shape = gradOut.shape.squeeze(0)
        }
    }

}