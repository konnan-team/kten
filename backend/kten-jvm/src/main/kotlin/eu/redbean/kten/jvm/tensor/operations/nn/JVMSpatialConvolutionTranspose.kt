package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.api.autograd.utils.squeeze
import eu.redbean.kten.api.autograd.utils.unsqueeze
import eu.redbean.kten.jvm.tensor.operations.AbstractJVMTensorOperations
import eu.redbean.kten.jvm.tensor.operations.JVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import java.util.stream.IntStream

class JVMSpatialConvolutionTranspose(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    dilationHeight: Int, dilationWidth: Int,
    private val outputPaddingHeight: Int, private val outputPaddingWidth: Int,
    ops: AbstractJVMTensorOperations
): JVMSpatialConvolutionBase(kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth, ops) {

    override val columnsShape: List<Int>
        get() = listOf(outputPlane * kernelHeight * kernelWidth, inputHeight * inputWidth)

    override fun setInputOutputPlanes(weight: JVMRawTensor) {
        inputPlane = weight.shape[0]
        outputPlane = weight.shape[1]
    }

    override fun calculateOutputHeightWidth() {
        if ((outputPaddingWidth >= strideWidth && outputPaddingWidth >= dilationWidth)
            || (outputPaddingHeight >= strideHeight && outputPaddingHeight >= dilationHeight))
            throw IllegalArgumentException("Output padding must be smaller than either stride or dilation")

        outputHeight = (inputHeight - 1) * strideHeight - 2 * paddingHeight + (dilationHeight * (kernelHeight - 1) + 1) + outputPaddingHeight
        outputWidth = (inputWidth - 1) * strideWidth - 2 * paddingWidth + (dilationWidth * (kernelWidth - 1) + 1) + outputPaddingWidth
    }

    override fun checkBiasShape(bias: JVMRawTensor?, weight: JVMRawTensor) {
        if (bias != null && (bias.dimensions != 1 || bias.shape[0] != weight.shape[1]))
            throw IllegalArgumentException(
                "Bias must have shape: [weight.shape[1]], " +
                        "but got bias with shape: ${bias.shape} and weight with shape: ${weight.shape}"
            )
    }

    override fun checkWeightDimensions(weight: JVMRawTensor) {
        if (weight.dimensions !in listOf(2, 4))
            throw IllegalArgumentException("Weight must be 2D or 4D tensor")
    }

    override fun calculateOutput(
        input: JVMRawTensor,
        weight: JVMRawTensor,
        bias: JVMRawTensor?,
    ): JVMRawTensor {
        shapeCheck(input, null, weight, bias)

        var batched = true

        if (input.dimensions == 3) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
        }

        // input shape = batch x inputChannels x H x W
        // weight shape = inputChannels x outputChannels x kernelH x kernelW

        val batchSize = input.shape[0]

        val output = JVMTensorOperations.createRaw(listOf(batchSize, outputPlane, outputHeight, outputWidth))

        resizeColumnsIfNeeded() // (outputChannels * kernelHeight * kernelWidth) x (inputHeight * inputWidth)

        resizeOnesIfNeeded()

        IntStream.range(0, batchSize).parallel().forEach {
            val columns = getLocalColumns()
            val inputView = input.getView(it)
            val outputView = output.getView(it)

            /**
             * weight is viewed as (inputChannels) x (outputChannels * kernelH * kernelW)
             * inputView is viewed as (inputChannels) x (inputHeight * inputWidth)
             *
             * columns is viewed as (outputChannels * kernelH * kernelW) x (inputHeight * inputWidth)
             *
             * calculates columns = 1 * (weight.T @ inputView) + 0 * gradWeight
             */
            JVMTensorOperations.gemmViews(
                columns.asView(),
                weight.view(weight.shape[0], -1),
                inputView.view(-1, columns.shape[1]),
                1f, 0f, true, false
            )

            mapper.col2im(columns.storeReference, outputPlane, outputHeight, outputWidth, inputHeight, inputWidth, outputView.storeReference)

            if (bias != null) {
                /**
                 * bias is viewed as (outputChannels) x (1)
                 * ones is viewed as (1) x (outputHeight * outputWidth)
                 *
                 * outputView is viewed as (outputChannels) x (outputHeight * outputWidth)
                 *
                 * calculates outputView = 1 * (bias @ ones) + 1 * outputView
                 */
                JVMTensorOperations.gemmViews(
                    outputView.view(-1, outputHeight*outputWidth),
                    bias.view(bias.shape[0], 1),
                    ones.view(1, outputHeight*outputWidth),
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
        weight: JVMRawTensor,
    ): JVMRawTensor {
        shapeCheck(input, gradOut, weight, null)

        var batched = true
        if (input.dimensions == 3) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
            gradOut.shape = gradOut.shape.unsqueeze(0)
        }

        // gradOut shape = batchSize, outputPlane, outputHeight, outputWidth
        // weight shape = inputChannels x outputChannels x kernelH x kernelW

        val batchSize = input.shape[0]

        val gradInput = JVMTensorOperations.createRaw(listOf(batchSize, inputPlane, inputHeight, inputWidth))

        resizeColumnsIfNeeded() // (outputChannels * kernelHeight * kernelWidth) x (inputHeight * inputWidth)

        IntStream.range(0, batchSize).parallel().forEach {
            val gradColumns = getLocalColumns()
            val gradInputView = gradInput.getView(it)
            val gradOutputView = gradOut.getView(it)

            mapper.im2col(gradOutputView.storeReference, outputPlane, outputHeight, outputWidth, gradColumns.storeReference)

            /**
             * weight is viewed as (inputChannels) x (outputChannels * kernelH * kernelW)
             * gradColumns is viewed as (outputChannels * kernelH * kernelW) x (inputHeight * inputWidth)
             *
             * gradInputView is viewed as (inputChannels) x (inputHeight * inputWidth)
             *
             * calculates gradInputView = 1 * (weight @ gradColumns) + 0 * gradInputView
             */
            JVMTensorOperations.gemmViews(
                gradInputView.view(-1, gradColumns.shape[1]),
                weight.view(weight.shape[0], -1),
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
        if (input.dimensions == 3) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
            gradOut.shape = gradOut.shape.unsqueeze(0)
        }

        resizeOnesIfNeeded()

        resizeColumnsIfNeeded() // (outputChannels * kernelHeight * kernelWidth) x (inputHeight * inputWidth)

        val batchSize = input.shape[0]
        val columns = getLocalColumns()

        for (b in 0 until batchSize) {
            val inputView = input.getView(b)
            val gradOutputView = gradOut.getView(b)

            mapper.im2col(gradOutputView.storeReference, outputPlane, outputHeight, outputWidth, columns.storeReference)

            /**
             * inputView is viewed as (inputChannels) x (inputHeight * inputWidth)
             * columns is viewed as (outputChannels * kernelH * kernelW) x (inputHeight * inputWidth)
             *
             * gradWeight is viewed as (inputChannels) x (outputChannels * kernelH * kernelW)
             *
             * calculates gradWeight = scale * (inputView @ columns.T) + 1 * gradWeight
             */
            JVMTensorOperations.gemmViews(
                gradWeight.view(gradWeight.shape[0], -1),
                inputView.view(-1, columns.shape[1]),
                columns.asView(),
                scale, 1f, false, true
            )

            if (gradBias != null) {
                /**
                 * gradOutputView is viewed as (outputChannels) x (outputHeight * outputWidth)
                 * ones is viewed as (outputHeight * outputWidth)
                 *
                 * gradBias is viewed as (outputChannels)
                 *
                 * calculates gradBias = scale * (gradOutputView @ ones) + 1 * gradBias
                 */
                JVMTensorOperations.gemvViews(
                    gradBias.asView(),
                    gradOutputView.view(gradBias.shape[0], -1),
                    ones.view(outputHeight*outputWidth),
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