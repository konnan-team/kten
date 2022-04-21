package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.api.tensor.operations.nn.BatchNormOperation
import eu.redbean.kten.jvm.tensor.operations.AbstractJVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import java.util.stream.IntStream
import kotlin.math.sqrt

class JVMBatchNorm(
    val axis: Int,
    val momentum: Float,
    val epsilon: Float,
    val training: Boolean,
    val ops: AbstractJVMTensorOperations
) : BatchNormOperation<JVMRawTensor> {

    private fun applyForAxis(elementSize: Int, elementIndexInAxis: Int, shape: List<Int>, op: (Int) -> Unit) {
        val dimensions = shape.size
        var offset = 1
        for (shapeIndex in axis + 1 until dimensions)
            offset *= shape[shapeIndex]

        for (counter in 0 until elementSize) {
            val preset = counter / offset
            var realIndex = counter % offset
            realIndex = preset * shape[axis] * offset + elementIndexInAxis * offset + realIndex
            op(realIndex)
        }
    }

    override fun calculateOutput(
        input: JVMRawTensor,
        runningMean: JVMRawTensor?,
        runningVar: JVMRawTensor?,
        gamma: JVMRawTensor?,
        beta: JVMRawTensor?
    ): BatchNormOperation.BatchNormOutputs<JVMRawTensor> {
        val output = ops.createRaw(input.shape.toList())
        val currentMean = ops.createRaw(listOf(input.shape[axis]))
        val currentStd = ops.createRaw(listOf(input.shape[axis]))

        val inputElementSize = input.storeReference.size
        val elementSizeAtSingleIndexForAxis = inputElementSize / input.shape[axis]

        IntStream.range(0, input.shape[axis]).parallel().forEach { index ->
            val mean: Float
            var stdev = 0f

            if (training) {
                var sum = 0.0f

                applyForAxis(elementSizeAtSingleIndexForAxis, index, input.shape) { sum += input.storeReference[it] }

                mean = (sum / elementSizeAtSingleIndexForAxis)
                currentMean.storeReference[index] = mean

                sum = 0.0f
                applyForAxis(elementSizeAtSingleIndexForAxis, index, input.shape) {
                    val inputMinusMean = input.storeReference[it] - mean
                    sum += inputMinusMean * inputMinusMean
                }

                if (sum != 0.0f || epsilon != 0f) {
                    stdev = (1.0f / sqrt(sum / elementSizeAtSingleIndexForAxis + epsilon))
                }

                currentStd.storeReference[index] = stdev

                if (runningMean != null && runningVar != null) {
                    runningMean.storeReference[index] = (momentum * mean) + ((1f - momentum) * runningMean.storeReference[index])
                    val unbiasedVar = sum / (elementSizeAtSingleIndexForAxis - 1)
                    runningVar.storeReference[index] = ((momentum * unbiasedVar) + ((1f - momentum) * runningVar.storeReference[index]))
                }
            } else {
                mean = runningMean!!.storeReference[index]
                stdev = (1.0f / sqrt((runningVar!!.storeReference[index] + epsilon)))
            }

            val g = if (gamma != null) gamma.storeReference[index] else 1f
            val b = if (beta != null) beta.storeReference[index] else 0f

            applyForAxis(elementSizeAtSingleIndexForAxis, index, input.shape) {
                output.storeReference[it] = (input.storeReference[it] - mean) * stdev * g + b
            }
        }

        return BatchNormOperation.BatchNormOutputs(
            output, currentMean, currentStd
        )
    }

    override fun calculateGrads(
        input: JVMRawTensor,
        runningMean: JVMRawTensor?,
        runningVar: JVMRawTensor?,
        currentMean: JVMRawTensor,
        currentStd: JVMRawTensor,
        gamma: JVMRawTensor?,
        gradOut: JVMRawTensor,
        inRequiresGrad: Boolean,
        gammaRequiresGrad: Boolean,
        betaRequiresGrad: Boolean
    ): BatchNormOperation.BatchNormGrads<JVMRawTensor> {
        if (inRequiresGrad.not() && gammaRequiresGrad.not() && betaRequiresGrad.not()) {
            return BatchNormOperation.BatchNormGrads(null, null, null)
        }

        val gradIn = if (inRequiresGrad) ops.createRaw(input.shape.toList()) else null
        val gradGamma = if (gammaRequiresGrad) ops.createRaw(listOf(input.shape[axis])) else null
        val gradBeta = if (betaRequiresGrad) ops.createRaw(listOf(input.shape[axis])) else null


        val inputElementSize = input.storeReference.size
        val elementSizeAtSingleIndexForAxis = inputElementSize / input.shape[axis]

        IntStream.range(0, input.shape[axis]).parallel().forEach { index ->
            val gammaAtIndex = if (gamma != null) gamma.storeReference[index] else 1f
            val mean: Float
            val stdev: Float

            if (training) {
                mean = currentMean.storeReference[index]
                stdev = currentStd.storeReference[index]
            } else {
                mean = runningMean!!.storeReference[index]
                stdev = (1.0 / sqrt((runningVar!!.storeReference[index] + epsilon).toDouble())).toFloat()
            }

            var sum = 0.0f
            applyForAxis(elementSizeAtSingleIndexForAxis, index, input.shape) {
                sum += gradOut.storeReference[it]
            }

            var dotProd = 0.0f
            applyForAxis(elementSizeAtSingleIndexForAxis, index, input.shape) {
                dotProd += (input.storeReference[it] - mean) * gradOut.storeReference[it]
            }

            if (gradIn != null) {
                if (training) {
                    val dotProdStdSq = (dotProd * stdev * stdev / elementSizeAtSingleIndexForAxis)
                    val gradMean = sum / elementSizeAtSingleIndexForAxis
                    applyForAxis(elementSizeAtSingleIndexForAxis, index, input.shape) {
                        val gradInBase = (input.storeReference[it] - mean) * dotProdStdSq
                        gradIn.storeReference[it] = ((gradOut.storeReference[it] - gradMean - gradInBase) * stdev * gammaAtIndex)
                    }
                } else {
                    applyForAxis(elementSizeAtSingleIndexForAxis, index, input.shape) {
                        gradIn.storeReference[it] = gradOut.storeReference[it] * stdev * gammaAtIndex
                    }
                }
            }

            if (gradGamma != null) {
                gradGamma.storeReference[index] = (dotProd * stdev)
            }

            if (gradBeta != null) {
                gradBeta.storeReference[index] = sum
            }
        }

        return BatchNormOperation.BatchNormGrads(
            gradIn, gradGamma, gradBeta
        )
    }

}