package eu.redbean.kten.api.autograd.functions.nn

import eu.redbean.kten.api.autograd.functions.Function
import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.autograd.tensor.NoGradVariable
import eu.redbean.kten.api.autograd.utils.normalizeAxis
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.operations.nn.BatchNormOperation
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

class BatchNormalizationFunction(
    val axis: Int,
    val momentum: Float,
    val epsilon: Float,
    val training: Boolean,
    ops: TensorOperations<AbstractRawTensor<Any>>
) : Function(ops) {

    private lateinit var input: Tensor
    private var runningMean: Tensor? = null
    private var runningVar: Tensor? = null
    private var gamma: Tensor? = null
    private var beta: Tensor? = null
    private lateinit var batchNormOp: BatchNormOperation<AbstractRawTensor<Any>>

    operator fun invoke(input: Tensor, runningMean: Tensor?, runningVar: Tensor?, gamma: Tensor?, beta: Tensor?): BatchNormalizationFunction {
        this.input = if (input is AGTensor) input.inplaceSafe() else input

        if (runningMean != null && runningVar == null) {
            throw IllegalArgumentException("Running mean and running var must be both set to non-null or both has to be null")
        }

        if (runningMean == null && runningVar == null && !training) {
            throw IllegalArgumentException("Running mean and running var must be non-null for inference mode")
        }

        if (runningMean != null) {
            if (runningMean !is NoGradVariable || runningVar !is NoGradVariable) {
                throw IllegalArgumentException("Running mean and running var parameters for batch normalization must be variables requiring no gradients")
            }

            this.runningMean = runningMean
            this.runningMean?.incrementRef()
            this.runningVar = runningVar
            this.runningVar?.incrementRef()
        }

        this.gamma = if (gamma is AGTensor) gamma.inplaceSafe() else gamma
        this.beta = if (beta is AGTensor) beta.inplaceSafe() else beta
        val normAxis = input.shape.normalizeAxis(axis)

        keepOutput = true

        val inputSizeAtAxis = this.input.shape[normAxis]

        if (runningMean != null && runningVar != null) {
            if (runningMean.dimensions != 1 || runningVar.dimensions != 1 || (this.gamma?.dimensions ?: 1) != 1 || (this.beta?.dimensions ?: 1) != 1
                || runningMean.shape[0] != inputSizeAtAxis || runningVar.shape[0] != inputSizeAtAxis
            ) {
                throw IllegalArgumentException(
                    "Running mean and running var parameters for batch normalization all must have " +
                            "shape: [$inputSizeAtAxis], but got tensors with shapes: " +
                            "running mean: ${runningMean.shape} " +
                            "running var: ${runningVar.shape} "
                )
            }
        }

        if ((this.gamma?.shape?.get(0) ?: inputSizeAtAxis) != inputSizeAtAxis || (this.beta?.shape?.get(0) ?: inputSizeAtAxis) != inputSizeAtAxis) {
            throw IllegalArgumentException(
                "Gamma and beta parameters for batch normalization all must have " +
                        "shape: [$inputSizeAtAxis], but got tensors with shapes: " +
                        "gamma: ${gamma?.shape} " +
                        "beta: ${beta?.shape}"
            )
        }

        this.batchNormOp = ops.batchNorm(normAxis, momentum, epsilon, training)
        this.cachedShape = input.shape.toList()
        return this
    }

    override fun getInputsAsList(): List<Tensor> {
        val inputsList = mutableListOf(input)
        if (gamma != null)
            inputsList.add(gamma!!)
        if (beta != null)
            inputsList.add(beta!!)
        return inputsList
    }

    override fun internalForward() {
        super.internalForward()
        if (hasValue()) {
            return
        }
        val inputFunctions = listOf(input, runningMean, runningVar, gamma, beta).filter { it is Function }.map { it as Function }
        inputFunctions.forEach(Function::internalForward)

        if (hasValue().not()) {
            doForward(input.getRawValue(), runningMean?.getRawValue(), runningVar?.getRawValue(), gamma?.getRawValue(), beta?.getRawValue())
            nanCheck(output!!)
        }

        inputFunctions.forEach(Function::mayFreeOutput)
    }

    private fun doForward(
        input: AbstractRawTensor<Any>,
        runningMean: AbstractRawTensor<Any>?,
        runningVar: AbstractRawTensor<Any>?,
        gamma: AbstractRawTensor<Any>?,
        beta: AbstractRawTensor<Any>?
    ) {
        val (output, currentMean, currentStd) = batchNormOp.calculateOutput(input, runningMean, runningVar, gamma, beta)
        if (gamma != null && runningMean != null && runningVar != null)
            saveForBackward(input, currentMean, currentStd, runningMean, runningVar, gamma)
        else
            if (gamma != null)
                saveForBackward(input, currentMean, currentStd, gamma)
            else
                if (runningMean != null && runningVar != null)
                    saveForBackward(input, currentMean, currentStd, runningMean, runningVar)
                else
                    saveForBackward(input, currentMean, currentStd)

        this.output = output
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input, currentMean, currentStd) = valuesSaved
        val gamma = if (valuesSaved.size == 4) valuesSaved[3] else if (valuesSaved.size == 6) valuesSaved[5] else null
        val runningMean = if (valuesSaved.size in 3..4) null else valuesSaved[3]
        val runningVar = if (valuesSaved.size in 3..4) null else valuesSaved[4]

        if (gradient.shape != input.shape) {
            throw IllegalStateException("Gradient shape mismatch, output grad shape: ${gradient.shape} input shape: ${input.shape}")
        }

        val (gradIn, gradGamma, gradBeta) = batchNormOp.calculateGrads(
            input, runningMean, runningVar, currentMean, currentStd, gamma, gradient,
            this.input.requiresGrad,
            this.gamma?.requiresGrad ?: false,
            this.beta?.requiresGrad ?: false
        )

        ops.release(currentMean, currentStd)

        return listOf(gradIn, gradGamma, gradBeta)
    }


}