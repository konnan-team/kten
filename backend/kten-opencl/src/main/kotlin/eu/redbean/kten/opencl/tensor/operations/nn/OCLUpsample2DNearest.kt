package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.nn.Upsample2DOperation
import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor

class OCLUpsample2DNearest(
    override val scale: Int,
    val ops: OCLTensorOperations
) : Upsample2DOperation<OCLRawTensor> {

    override fun upsample(input: OCLRawTensor): OCLRawTensor {
        checkDimensions(input.shape)
        val inputTensor = if (input.dimensions == 3) input.view(listOf(1) + input.shape) else input
        val outputShape = calculateOutputShape(inputTensor.shape)
        val outputTensor = ops.createRaw(outputShape)

        ops.environment.kernelStore.upsampleNearestUpdateOutput(
            inputTensor.storeReference,
            outputTensor.storeReference,
            outputShape,
            scale
        )

        if (input.dimensions == 3)
            outputTensor.inplaceReshape(outputTensor.shape.drop(1))

        return outputTensor
    }

    override fun calculateGrad(gradOut: OCLRawTensor, inputShape: List<Int>): OCLRawTensor {
        checkDimensions(gradOut.shape)
        checkGradOutShape(gradOut.shape, inputShape)
        val gradOutTensor = if (gradOut.dimensions == 3) gradOut.view(listOf(1) + gradOut.shape) else gradOut
        val gradInShape = if (inputShape.size == 3) listOf(1) + inputShape else inputShape
        val gradInTensor = ops.createRawFill(gradInShape, 0f)

        ops.environment.kernelStore.upsampleNearestUpdateGrad(
            gradInTensor.storeReference,
            gradOutTensor.storeReference,
            gradInShape,
            scale
        )

        if (inputShape.size == 3)
            gradInTensor.inplaceReshape(inputShape)

        return gradInTensor
    }

}