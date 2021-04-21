package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.api.autograd.utils.squeeze
import eu.redbean.kten.api.autograd.utils.unsqueeze
import eu.redbean.kten.jvm.tensor.operations.AbstractJVMTensorOperations
import eu.redbean.kten.jvm.tensor.operations.JVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import java.util.stream.IntStream

class JVMSpatialConvolution(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    dilationHeight: Int, dilationWidth: Int,
    ops: AbstractJVMTensorOperations
): JVMSpatialConvolutionBase(kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth, ops) {

    override val columnsShape: List<Int>
        get() = listOf(inputPlane * kernelHeight * kernelWidth, outputHeight * outputWidth)

    override fun checkWeightDimensions(weight: JVMRawTensor) {
        if (weight.dimensions != 4)
            throw IllegalArgumentException("Weight must be 4D tensor")
    }

    override fun checkBiasShape(bias: JVMRawTensor?, weight: JVMRawTensor) {
        if (bias != null && (bias.dimensions != 1 || bias.shape[0] != weight.shape[0]))
            throw IllegalArgumentException(
                "Bias must have shape: [weight.shape[0]], " +
                        "but got bias with shape: ${bias.shape} and weight with shape: ${weight.shape}"
            )
    }

    override fun calculateOutputHeightWidth() {
        outputHeight = (inputHeight + 2 * paddingHeight - (dilationHeight * (kernelHeight - 1) + 1)) / strideHeight + 1
        outputWidth = (inputWidth + 2 * paddingWidth - (dilationWidth * (kernelWidth - 1) + 1)) / strideWidth + 1
    }

    override fun setInputOutputPlanes(weight: JVMRawTensor) {
        inputPlane = weight.shape[1]
        outputPlane = weight.shape[0]
    }

    override fun calculateOutput(
        input: JVMRawTensor,
        weight: JVMRawTensor,
        bias: JVMRawTensor?
    ): JVMRawTensor {
        shapeCheck(input, null, weight, bias)

        var batched = true
        if (input.dimensions == 3) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
        }

        val batchSize = input.shape[0]

        val output = JVMTensorOperations.createRaw(listOf(batchSize, outputPlane, outputHeight, outputWidth))

        resizeColumnsIfNeeded()
        resizeOnesIfNeeded()

        IntStream.range(0, batchSize).parallel().forEach {
            val localColumns = getLocalColumns()

            val inputView = input.getView(it)
            val outputView = output.getView(it)

            if (bias != null) {
                JVMTensorOperations.gemmViews(
                    outputView.view(outputPlane, outputHeight * outputWidth),
                    bias.view(outputPlane, 1),
                    this.ones.view(1, outputHeight * outputWidth),
                    1f, 0f, false, false
                )
            }

            mapper.im2col(inputView.storeReference, inputPlane, inputHeight, inputWidth, localColumns.storeReference)

            JVMTensorOperations.gemmViews(
                outputView.view(outputPlane, outputHeight * outputWidth),
                weight.view(outputPlane, inputPlane * kernelHeight * kernelWidth),
                localColumns.asView(),
                1f, 1f, false, false
            )

            columnsPool.push(localColumns)
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
        if (input.dimensions == 3) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
            gradOut.shape = gradOut.shape.unsqueeze(0)
        }

        val batchSize = input.shape[0]

        val gradInput = JVMTensorOperations.createRaw(listOf(batchSize, inputPlane, inputHeight, inputWidth))

        resizeColumnsIfNeeded()

        IntStream.range(0, batchSize).parallel().forEach {
            val columnsLocal = getLocalColumns()
            val gradInputView = gradInput.getView(it)
            val gradOutputView = gradOut.getView(it)

            JVMTensorOperations.gemmViews(
                columnsLocal.asView(),
                weight.view(outputPlane, inputPlane * kernelHeight * kernelWidth),
                gradOutputView.view(outputPlane, outputHeight * outputWidth),
                1f, 0f, true, false
            )

            mapper.col2im(columnsLocal.storeReference, inputPlane, inputHeight, inputWidth, outputHeight, outputWidth, gradInputView.storeReference)
            columnsPool.push(columnsLocal)
        }

        if (!batched) {
            input.shape = input.shape.squeeze(0)
            gradOut.shape = gradOut.shape.squeeze(0)
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

        val batchSize = input.shape[0]

        resizeOnesIfNeeded()
        resizeColumnsIfNeeded()

        val localColumns = getLocalColumns()
        // cannot be parallelized because grads are accumulated in gemm and gemv
        IntStream.range(0, batchSize).forEach {
            val inputView = input.getView(it)
            val gradOutputView = gradOut.getView(it)

            mapper.im2col(inputView.storeReference, inputPlane, inputHeight, inputWidth, localColumns.storeReference)

            JVMTensorOperations.gemmViews(
                gradWeight.view(outputPlane, inputPlane * kernelHeight * kernelWidth),
                gradOutputView.view(outputPlane, outputHeight * outputWidth),
                localColumns.asView(),
                scale, 1f, false, true
            )

            if (gradBias != null) {
                JVMTensorOperations.gemvViews(
                    gradBias.asView(),
                    gradOutputView.view(outputPlane, outputHeight * outputWidth),
                    this.ones.view(outputHeight * outputWidth),
                    scale, 1f, false
                )
            }

        }
        columnsPool.push(localColumns)

        if (!batched) {
            input.shape = input.shape.squeeze(0)
            gradOut.shape = gradOut.shape.squeeze(0)
        }
    }
}