package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.api.autograd.utils.squeeze
import eu.redbean.kten.api.autograd.utils.unsqueeze
import eu.redbean.kten.api.tensor.operations.nn.ConvolutionOperation
import eu.redbean.kten.jvm.tensor.operations.AbstractJVMTensorOperations
import eu.redbean.kten.jvm.tensor.operations.JVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import java.util.stream.IntStream

class JVMVolumetricConvolutionTranspose(
    kernelDepth: Int, kernelHeight: Int, kernelWidth: Int,
    paddingDepth: Int, paddingHeight: Int, paddingWidth: Int,
    strideDepth: Int,strideHeight: Int, strideWidth: Int,
    dilationDepth: Int, dilationHeight: Int, dilationWidth: Int,
    private val outputPaddingDepth: Int, private val outputPaddingHeight: Int, private val outputPaddingWidth: Int,
    ops: AbstractJVMTensorOperations
): JVMVolumetricConvolutionBase(
    kernelDepth, kernelHeight, kernelWidth,
    paddingDepth, paddingHeight, paddingWidth,
    strideDepth, strideHeight, strideWidth,
    dilationDepth, dilationHeight, dilationWidth,
    ops
) {
    override val columnsShape: List<Int>
        get() = listOf(outputPlane * kernelDepth * kernelHeight * kernelWidth, inputDepth * inputHeight * inputWidth)

    override fun checkBiasShape(bias: JVMRawTensor?, weight: JVMRawTensor) {
        if (bias != null && (bias.dimensions != 1 || bias.shape[0] != weight.shape[1]))
            throw IllegalArgumentException(
                "Bias must have shape: [weight.shape[1]], " +
                        "but got bias with shape: ${bias.shape} and weight with shape: ${weight.shape}"
            )
    }

    override fun calculateOutputDepthHeightWidth() {
        if ((outputPaddingDepth >= strideDepth && outputPaddingDepth >= dilationDepth)
            || (outputPaddingWidth >= strideWidth && outputPaddingWidth >= dilationWidth)
            || (outputPaddingHeight >= strideHeight && outputPaddingHeight >= dilationHeight))
            throw IllegalArgumentException("Output padding must be smaller than either stride or dilation")

        outputDepth = (inputDepth - 1) * strideDepth - 2 * paddingDepth + (dilationDepth * (kernelDepth - 1) + 1) + outputPaddingDepth
        outputHeight = (inputHeight - 1) * strideHeight - 2 * paddingHeight + (dilationHeight * (kernelHeight - 1) + 1) + outputPaddingHeight
        outputWidth = (inputWidth - 1) * strideWidth - 2 * paddingWidth + (dilationWidth * (kernelWidth - 1) + 1) + outputPaddingWidth
    }

    override fun setInputOutputPlanes(weight: JVMRawTensor) {
        inputPlane = weight.shape[0]
        outputPlane = weight.shape[1]
    }


    override fun calculateOutput(
        input: JVMRawTensor,
        weight: JVMRawTensor,
        bias: JVMRawTensor?
    ): JVMRawTensor {
        shapeCheck(input, null, weight, bias)

        var batched = true

        if (input.dimensions == 4) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
        }

        val batchSize = input.shape[0]
        val output = JVMTensorOperations.createRaw(listOf(batchSize, outputPlane, outputDepth, outputHeight, outputWidth))

        resizeColumnsIfNeeded()
        resizeOnesIfNeeded()

        IntStream.range(0, batchSize).parallel().forEach {
            val columns = getLocalColumns()
            val inputView = input.getView(it)
            val outputView = output.getView(it)

            JVMTensorOperations.gemmViews(
                columns.asView(),
                weight.view(inputPlane, (outputPlane * kernelDepth * kernelHeight * kernelWidth)),
                inputView.view(inputPlane, (inputDepth * inputHeight * inputWidth)),
                1f, 0f, true, false
            )

            mapper.col2vol(
                columns.storeReference,
                outputPlane, outputDepth, outputHeight, outputWidth,
                inputDepth, inputHeight, inputWidth,
                outputView.storeReference
            )

            if (bias != null) {
                JVMTensorOperations.gemmViews(
                    outputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                    bias.view(bias.shape[0], 1),
                    ones.view(1, (outputDepth * outputHeight * outputWidth)),
                    1f, 1f, false, false
                )
            }

            columnsPool.push(columns)
        }

        if (!batched) {
            input.shape = input.shape.squeeze(0)
            output.shape = output.shape.squeeze(0)
        }

        return output
    }

    override fun calculateGradInput(
        input: JVMRawTensor,
        gradOut: JVMRawTensor,
        weight: JVMRawTensor
    ): JVMRawTensor {
        shapeCheck(input, gradOut, weight, null)

        var batched = true
        if (input.dimensions == 4) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
            gradOut.shape = gradOut.shape.unsqueeze(0)
        }

        val batchSize = input.shape[0]
        val gradInput = JVMTensorOperations.createRaw(listOf(batchSize, inputPlane, inputDepth, inputHeight, inputWidth))

        resizeColumnsIfNeeded()

        IntStream.range(0, batchSize).parallel().forEach {
            val gradColumns = getLocalColumns()
            val gradInputView = gradInput.getView(it)
            val gradOutputView = gradOut.getView(it)

            mapper.vol2col(
                gradOutputView.storeReference,
                outputPlane, outputDepth, outputHeight, outputWidth,
                gradColumns.storeReference
            )

            JVMTensorOperations.gemmViews(
                gradInputView.view(inputPlane, (inputDepth * inputHeight * inputWidth)),
                weight.view(inputPlane, (outputPlane * kernelDepth * kernelHeight * kernelWidth)),
                gradColumns.asView(),
                1f, 0f, false, false
            )

            columnsPool.push(gradColumns)
        }

        if (!batched) {
            gradOut.shape = gradOut.shape.squeeze(0)
            input.shape = input.shape.squeeze(0)
            gradInput.shape = gradInput.shape.squeeze(0)
        }

        return gradInput
    }

    override fun accumulateGradParams(
        input: JVMRawTensor,
        gradOut: JVMRawTensor,
        gradWeight: JVMRawTensor,
        gradBias: JVMRawTensor?,
        scale: Float
    ) {
        shapeCheck(input, gradOut, gradWeight, gradBias)

        var batched = true
        if (input.dimensions == 4) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
            gradOut.shape = gradOut.shape.unsqueeze(0)
        }

        resizeOnesIfNeeded()
        resizeColumnsIfNeeded()

        val batchSize = input.shape[0]
        val columns = getLocalColumns()

        for (b in 0 until batchSize) {
            val inputView = input.getView(b)
            val gradOutputView = gradOut.getView(b)

            mapper.vol2col(
                gradOutputView.storeReference,
                outputPlane, outputDepth, outputHeight, outputWidth,
                columns.storeReference
            )

            JVMTensorOperations.gemmViews(
                gradWeight.view(inputPlane, (outputPlane * kernelDepth * kernelHeight * kernelWidth)),
                inputView.view(inputPlane, (inputDepth * inputHeight * inputWidth)),
                columns.asView(),
                scale, 1f, false, true
            )

            if (gradBias != null) {
                JVMTensorOperations.gemvViews(
                    gradBias.asView(),
                    gradOutputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                    ones.view(outputDepth * outputHeight * outputWidth),
                    scale, 1f, false
                )
            }
        }

        columnsPool.push(columns)

        if (!batched) {
            input.shape = input.shape.squeeze(0)
            gradOut.shape = gradOut.shape.squeeze(0)
        }
    }

}