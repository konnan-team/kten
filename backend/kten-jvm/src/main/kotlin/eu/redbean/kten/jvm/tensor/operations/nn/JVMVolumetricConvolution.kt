package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.api.autograd.utils.squeeze
import eu.redbean.kten.api.autograd.utils.unsqueeze
import eu.redbean.kten.api.tensor.operations.nn.ConvolutionOperation
import eu.redbean.kten.jvm.tensor.operations.AbstractJVMTensorOperations
import eu.redbean.kten.jvm.tensor.operations.JVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import java.util.stream.IntStream

class JVMVolumetricConvolution(
    kernelDepth: Int, kernelHeight: Int, kernelWidth: Int,
    paddingDepth: Int, paddingHeight: Int, paddingWidth: Int,
    strideDepth: Int,strideHeight: Int, strideWidth: Int,
    dilationDepth: Int, dilationHeight: Int, dilationWidth: Int,
    ops: AbstractJVMTensorOperations
): JVMVolumetricConvolutionBase(
    kernelDepth, kernelHeight, kernelWidth,
    paddingDepth, paddingHeight, paddingWidth,
    strideDepth, strideHeight, strideWidth,
    dilationDepth, dilationHeight, dilationWidth,
    ops
) {
    override val columnsShape: List<Int>
        get() = listOf((inputPlane * kernelDepth * kernelHeight * kernelWidth), (outputDepth * outputHeight * outputWidth))

    override fun checkBiasShape(bias: JVMRawTensor?, weight: JVMRawTensor) {
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
        if (input.dimensions == 4) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
        }

        val batchSize = input.shape[0]

        val output = JVMTensorOperations.createRaw(listOf(batchSize, outputPlane, outputDepth, outputHeight, outputWidth))

        resizeColumnsIfNeeded()
        resizeOnesIfNeeded()

        IntStream.range(0, batchSize).parallel().forEach {
            val localColumns = getLocalColumns()

            val inputView = input.getView(it)
            val outputView = output.getView(it)

            if (bias != null) {
                JVMTensorOperations.gemmViews(
                    outputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                    bias.view(outputPlane, 1),
                    this.ones.view(1, outputDepth * outputHeight * outputWidth),
                    1f, 0f, false, false
                )
            }

            mapper.vol2col(
                inputView.storeReference,
                inputPlane, inputDepth, inputHeight, inputWidth,
                localColumns.storeReference
            )

            JVMTensorOperations.gemmViews(
                outputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                weight.view(outputPlane, (inputPlane * kernelDepth * kernelHeight * kernelWidth)),
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
        if (input.dimensions == 4) {
            batched = false
            input.shape = input.shape.unsqueeze(0)
            gradOut.shape = gradOut.shape.unsqueeze(0)
        }

        val batchSize = input.shape[0]

        val gradInput = JVMTensorOperations.createRaw(listOf(batchSize, inputPlane, inputDepth, inputHeight, inputWidth))

        resizeColumnsIfNeeded()

        IntStream.range(0, batchSize).parallel().forEach {
            val columnsLocal = getLocalColumns()
            val gradInputView = gradInput.getView(it)
            val gradOutputView = gradOut.getView(it)

            JVMTensorOperations.gemmViews(
                columnsLocal.asView(),
                weight.view(outputPlane, (inputPlane * kernelDepth * kernelHeight * kernelWidth)),
                gradOutputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                1f, 0f, true, false
            )

            mapper.col2vol(
                columnsLocal.storeReference,
                inputPlane, inputDepth, inputHeight, inputWidth,
                outputDepth, outputHeight, outputWidth,
                gradInputView.storeReference
            )

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
        if (input.dimensions == 4) {
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

            mapper.vol2col(
                inputView.storeReference,
                inputPlane, inputDepth, inputHeight, inputWidth,
                localColumns.storeReference
            )

            JVMTensorOperations.gemmViews(
                gradWeight.view(outputPlane, (inputPlane * kernelDepth * kernelHeight * kernelWidth)),
                gradOutputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                localColumns.asView(),
                scale, 1f, false, true
            )

            if (gradBias != null) {
                JVMTensorOperations.gemvViews(
                    gradBias.asView(),
                    gradOutputView.view(outputPlane, (outputDepth * outputHeight * outputWidth)),
                    this.ones.view(outputDepth * outputHeight * outputWidth),
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