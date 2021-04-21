package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.api.autograd.utils.squeeze
import eu.redbean.kten.api.autograd.utils.unsqueeze
import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor

class OCLVolumetricConvolution(
    kernelDepth: Int, kernelHeight: Int, kernelWidth: Int,
    paddingDepth: Int, paddingHeight: Int, paddingWidth: Int,
    strideDepth: Int,strideHeight: Int, strideWidth: Int,
    dilationDepth: Int, dilationHeight: Int, dilationWidth: Int,
    ops: OCLTensorOperations
): OCLVolumetricConvolutionBase(
    kernelDepth, kernelHeight, kernelWidth,
    paddingDepth, paddingHeight, paddingWidth,
    strideDepth, strideHeight, strideWidth,
    dilationDepth, dilationHeight, dilationWidth,
    ops
) {
    override val columnsShape: List<Int>
        get() = listOf((inputPlane * kernelDepth * kernelHeight * kernelWidth), (outputDepth * outputHeight * outputWidth))

    override fun checkBiasShape(bias: OCLRawTensor?, weight: OCLRawTensor) {
        if (bias != null && (bias.dimensions != 1 || bias.shape[0] != weight.shape[0]))
            throw IllegalArgumentException(
                "Bias must have shape: [weight.shape[0]], " +
                        "but got bias with shape: ${bias.shape} and weight with shape: ${weight.shape}"
            )
    }

    override fun calculateOutputDepthHeightWidth() {
        outputDepth = (inputDepth + 2 * paddingDepth - (dilationDepth * (kernelDepth - 1) + 1)) / strideDepth + 1
        outputHeight = (inputHeight + 2 * paddingHeight - (dilationHeight * (kernelHeight - 1) + 1)) / strideHeight + 1
        outputWidth = (inputWidth + 2 * paddingWidth - (dilationWidth * (kernelWidth - 1) + 1)) / strideWidth + 1
    }

    override fun setInputOutputPlanes(weight: OCLRawTensor) {
        inputPlane = weight.shape[1]
        outputPlane = weight.shape[0]
    }

    override fun calculateOutput(input: OCLRawTensor, weight: OCLRawTensor, bias: OCLRawTensor?): OCLRawTensor {
        shapeCheck(input, null, weight, bias)

        var batched = true
        if (input.dimensions == 4) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
        }

        val batchSize = input.shape[0]

        val output = ops.createRaw(listOf(batchSize, outputPlane, outputDepth, outputHeight, outputWidth))

        resizeOnesIfNeeded()

        val columns = ops.createRaw(columnsShape)

        for (b in 0 until batchSize) {
            val inputView = input.getView(b)
            val outputView = output.getView(b)

            if (bias != null) {
                ops.gemmViews(
                    outputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                    bias.view(outputPlane, 1),
                    this.ones.view(1, outputDepth * outputHeight * outputWidth),
                    1f, 0f, false, false
                )
            }

            mapper.vol2col(
                inputView.storeReference,
                inputPlane, inputDepth, inputHeight, inputWidth,
                columns.storeReference
            )

            ops.gemmViews(
                outputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                weight.view(outputPlane, (inputPlane * kernelDepth * kernelHeight * kernelWidth)),
                columns.asView(),
                1f, 1f, false, false
            )
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
        if (input.dimensions == 4) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
            gradOut.shape = gradOut.shape.unsqueeze(0)
        }

        val batchSize = input.shape[0]

        val gradInput = ops.createRaw(listOf(batchSize, inputPlane, inputDepth, inputHeight, inputWidth))

        val columns = ops.createRaw(columnsShape)

        for (b in 0 until batchSize) {
            val gradInputView = gradInput.getView(b)
            val gradOutputView = gradOut.getView(b)

            ops.gemmViews(
                columns.asView(),
                weight.view(outputPlane, (inputPlane * kernelDepth * kernelHeight * kernelWidth)),
                gradOutputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                1f, 0f, true, false
            )

            mapper.col2vol(
                columns.storeReference,
                inputPlane, inputDepth, inputHeight, inputWidth,
                outputDepth, outputHeight, outputWidth,
                gradInputView.storeReference
            )
        }

        if (!batched) {
            input.shape = input.shape.squeeze(0)
            gradOut.shape = gradOut.shape.squeeze(0)
            gradInput.shape = gradInput.shape.squeeze(0)
        }

        return gradInput
    }

    override fun accumulateGradParams(input: OCLRawTensor, gradOut: OCLRawTensor, gradWeight: OCLRawTensor, gradBias: OCLRawTensor?, scale: Float) {
        shapeCheck(input, gradOut, gradWeight, gradBias)

        var batched = true
        if (input.dimensions == 4) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
            gradOut.shape = gradOut.shape.unsqueeze(0)
        }

        val batchSize = input.shape[0]

        resizeOnesIfNeeded()

        val columns = ops.createRaw(columnsShape)

        for (b in 0 until batchSize) {
            val inputView = input.getView(b)
            val gradOutputView = gradOut.getView(b)

            mapper.vol2col(
                inputView.storeReference,
                inputPlane, inputDepth, inputHeight, inputWidth,
                columns.storeReference
            )

            ops.gemmViews(
                gradWeight.view(outputPlane, (inputPlane * kernelDepth * kernelHeight * kernelWidth)),
                gradOutputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                columns.asView(),
                scale, 1f, false, true
            )

            if (gradBias != null) {
                ops.gemvViews(
                    gradBias.asView(),
                    gradOutputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                    this.ones.view(outputDepth * outputHeight * outputWidth),
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