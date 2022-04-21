package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.nn.PoolingResult
import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor

class OCLSpatialMaxPooling(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    dilationHeight: Int, dilationWidth: Int,
    useCeil: Boolean,
    ops: OCLTensorOperations
) : OCLSpatialPoolingBase(
    kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth, useCeil, ops
) {

    override fun updateOutput(input: OCLRawTensor): PoolingResult<OCLRawTensor> {
        shapeCheck(input, null, null)

        val res = if (input.dimensions == 3)
            PoolingResult(
                ops.createRaw(listOf(outputPlane, outputHeight, outputWidth)),
                ops.createRaw(listOf(outputPlane, outputHeight, outputWidth))
            )
        else
            PoolingResult(
                ops.createRaw(listOf(input.shape[0], outputPlane, outputHeight, outputWidth)),
                ops.createRaw(listOf(input.shape[0], outputPlane, outputHeight, outputWidth))
            )

        ops.environment.kernelStore.maxPoolingUpdateOutput(
            input.storeReference,
            res.output.storeReference,
            res.indices!!.storeReference,
            input.shape,
            res.output.shape,
            kernelHeight, kernelWidth,
            strideHeight, strideWidth,
            paddingHeight, paddingWidth,
            dilationHeight, dilationWidth
        )

        return res
    }

    override fun calculateGradInput(input: OCLRawTensor, gradOut: OCLRawTensor, indices: OCLRawTensor?): OCLRawTensor {
        shapeCheck(input, gradOut, indices)

        val gradIn = ops.createRawFill(input.shape, 0f)

        ops.environment.kernelStore.maxPoolingUpdateGradIn(
            gradIn.storeReference,
            gradOut.storeReference,
            indices!!.storeReference,
            gradIn.shape,
            gradOut.shape
        )

        return gradIn
    }

}