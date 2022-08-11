package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.nn.BatchNormOperation
import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor

class OCLBatchNorm(
    val axis: Int,
    val momentum: Float,
    val epsilon: Float,
    val training: Boolean,
    val ops: OCLTensorOperations
) : BatchNormOperation<OCLRawTensor> {

    override fun calculateOutput(
        input: OCLRawTensor,
        runningMean: OCLRawTensor?,
        runningVar: OCLRawTensor?,
        gamma: OCLRawTensor?,
        beta: OCLRawTensor?
    ): BatchNormOperation.BatchNormOutputs<OCLRawTensor> {
        val output = ops.createRaw(input.shape.toList())
        val currentMean = ops.createRaw(listOf(input.shape[axis]))
        val currentStd = ops.createRaw(listOf(input.shape[axis]))
        ops.environment.kernelStore.batchNormUpdateOutput(
            input.storeReference,
            output.storeReference,
            gamma?.storeReference,
            beta?.storeReference,
            runningMean?.storeReference,
            runningVar?.storeReference,
            currentMean.storeReference,
            currentStd.storeReference,
            input.shape,
            axis,
            epsilon,
            momentum,
            training
        )

        return BatchNormOperation.BatchNormOutputs(
            output, currentMean, currentStd
        )
    }

    override fun calculateGrads(
        input: OCLRawTensor,
        runningMean: OCLRawTensor?,
        runningVar: OCLRawTensor?,
        currentMean: OCLRawTensor,
        currentStd: OCLRawTensor,
        gamma: OCLRawTensor?,
        gradOut: OCLRawTensor,
        inRequiresGrad: Boolean,
        gammaRequiresGrad: Boolean,
        betaRequiresGrad: Boolean
    ): BatchNormOperation.BatchNormGrads<OCLRawTensor> {
        if (inRequiresGrad.not() && gammaRequiresGrad.not() && betaRequiresGrad.not()) {
            return BatchNormOperation.BatchNormGrads(null, null, null)
        }

        val gradIn = if (inRequiresGrad) ops.createRaw(input.shape.toList()) else null
        val gradGamma = if (gammaRequiresGrad) ops.createRaw(listOf(input.shape[axis])) else null
        val gradBeta = if (betaRequiresGrad) ops.createRaw(listOf(input.shape[axis])) else null

        ops.environment.kernelStore.batchNormUpdateGrads(
            input.storeReference,
            gradOut.storeReference,
            gamma?.storeReference,
            runningMean?.storeReference,
            runningVar?.storeReference,
            currentMean.storeReference,
            currentStd.storeReference,
            gradIn?.storeReference,
            gradGamma?.storeReference,
            gradBeta?.storeReference,
            input.shape,
            axis,
            epsilon,
            training
        )

        return BatchNormOperation.BatchNormGrads(
            gradIn, gradGamma, gradBeta
        )
    }
}