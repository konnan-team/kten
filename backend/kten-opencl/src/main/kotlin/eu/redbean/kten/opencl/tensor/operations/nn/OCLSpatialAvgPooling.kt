package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.nn.PoolingResult
import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor

class OCLSpatialAvgPooling(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    useCeil: Boolean,
    val includePadding: Boolean,
    ops: OCLTensorOperations
) : OCLSpatialPoolingBase(
    kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, 1, 1, useCeil, ops
) {

    override fun updateOutput(input: OCLRawTensor): PoolingResult<OCLRawTensor> {
        shapeCheck(input, null, null)

        val output = if (input.dimensions == 3)
            ops.createRaw(listOf(outputPlane, outputHeight, outputWidth))
        else
            ops.createRaw(listOf(input.shape[0], outputPlane, outputHeight, outputWidth))

        ops.environment.kernelStore.avgPoolingUpdateOutput(
            input.storeReference,
            output.storeReference,
            input.shape,
            output.shape,
            kernelHeight, kernelWidth,
            strideHeight, strideWidth,
            paddingHeight, paddingWidth,
            includePadding
        )

        return PoolingResult(output, null)
    }

    override fun calculateGradInput(input: OCLRawTensor, gradOut: OCLRawTensor, indices: OCLRawTensor?): OCLRawTensor {
        shapeCheck(input, gradOut, null)

        val gradIn = ops.createRawFill(input.shape, 0f)

        ops.environment.kernelStore.avgPoolingUpdateGradIn(
            gradIn.storeReference,
            gradOut.storeReference,
            gradIn.shape,
            gradOut.shape,
            kernelHeight, kernelWidth,
            strideHeight, strideWidth,
            paddingHeight, paddingWidth,
            includePadding
        )

        return gradIn
    }

}