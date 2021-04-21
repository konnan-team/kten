package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.api.autograd.utils.squeeze
import eu.redbean.kten.api.autograd.utils.unsqueeze
import eu.redbean.kten.jvm.tensor.operations.JVMTensorOperations
import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.MemoryAccessOption.SOURCE
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.MemoryAccessOption.TARGET
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor
import org.jocl.blast.CLBlast
import org.jocl.blast.CLBlastKernelMode

class OCLSpatialConvolution(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    dilationHeight: Int, dilationWidth: Int,
    ops: OCLTensorOperations
): OCLSpatialConvolutionBase(
    kernelHeight, kernelWidth,
    paddingHeight, paddingWidth,
    strideHeight, strideWidth,
    dilationHeight, dilationWidth,
    ops
) {

    override val columnsShape: List<Int>
        get() = listOf(inputPlane * kernelHeight * kernelWidth, outputHeight * outputWidth)

    override fun checkWeightDimensions(weight: OCLRawTensor) {
        if (weight.dimensions != 4)
            throw IllegalArgumentException("Weight must be 4D tensor")
    }

    override fun checkBiasShape(bias: OCLRawTensor?, weight: OCLRawTensor) {
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

    override fun setInputOutputPlanes(weight: OCLRawTensor) {
        inputPlane = weight.shape[1]
        outputPlane = weight.shape[0]
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

        CLBlast.CLBlastSconvgemm(
            CLBlastKernelMode.CLBlastKernelModeCrossCorrelation, //TODO test non symmetric kernels, to make sure this is the right option
            inputPlane.toLong(), inputHeight.toLong(), inputWidth.toLong(),
            kernelHeight.toLong(), kernelWidth.toLong(),
            paddingHeight.toLong(), paddingWidth.toLong(),
            strideHeight.toLong(), strideWidth.toLong(),
            dilationHeight.toLong(), dilationWidth.toLong(),
            outputPlane.toLong(), batchSize.toLong(),
            input.storeReference.getMemoryObject(SOURCE), 0L,
            weight.storeReference.getMemoryObject(SOURCE), 0L,
            output.storeReference.getMemoryObject(TARGET), 0L,
            ops.environment.commandQueue,
            null
        )

        if (bias != null) {
            bias.shape = listOf(1, outputPlane, 1, 1)
            val expandedBias = bias.broadcastTo(output.shape)
            output += expandedBias
            bias.shape = listOf(outputPlane)
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

        val columns = ops.createRaw(columnsShape)

        for (b in 0 until batchSize) {
            val gradInputView = gradInput.getView(b)
            val gradOutputView = gradOut.getView(b)

            ops.gemmViews(
                columns.asView(),
                weight.view(outputPlane, inputPlane * kernelHeight * kernelWidth),
                gradOutputView.view(outputPlane, outputHeight * outputWidth),
                1f, 0f, true, false
            )

            mapper.col2im(columns.storeReference, inputPlane, inputHeight, inputWidth, gradInputView.storeReference)
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
        if (input.dimensions == 3) {
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

            mapper.im2col(inputView.storeReference, inputPlane, inputHeight, inputWidth, columns.storeReference)

            ops.gemmViews(
                gradWeight.view(outputPlane, inputPlane * kernelHeight * kernelWidth),
                gradOutputView.view(outputPlane, outputHeight * outputWidth),
                columns.asView(),
                scale, 1f, false, true
            )

            if (gradBias != null) {
                ops.gemvViews(
                    gradBias.asView(),
                    gradOutputView.view(outputPlane, outputHeight * outputWidth),
                    this.ones.view(outputHeight * outputWidth),
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