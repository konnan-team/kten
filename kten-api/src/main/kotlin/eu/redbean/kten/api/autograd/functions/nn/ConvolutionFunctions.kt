package eu.redbean.kten.api.autograd.functions.nn

import eu.redbean.kten.api.autograd.functions.Function
import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.operations.nn.ConvolutionOperation
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

abstract class AbstractConvND(
    ops: TensorOperations<AbstractRawTensor<Any>>
): Function(ops) {

    protected lateinit var input: Tensor
    protected lateinit var weight: Tensor
    protected var bias: Tensor? = null

    operator fun invoke(input: Tensor, weight: Tensor, bias: Tensor?): AbstractConvND {
        this.input = if (input is AGTensor) input.inplaceSafe() else input
        this.weight = if (weight is AGTensor) weight.inplaceSafe() else weight
        this.bias = if (bias is AGTensor) bias.inplaceSafe() else bias
        this.cachedShape = calculateOutputShape()
        return this
    }

    abstract fun calculateOutputShape(): List<Int>

    override fun getInputsAsList(): List<Tensor> {
        if (bias != null)
            return listOf(input, weight, bias!!)
        return listOf(input, weight)
    }

    override fun internalForward() {
        super.internalForward()
        if (hasValue()) {
            return
        }
        val inputFunctions = listOf(input, weight, bias).filter { it is Function }.map { it as Function }
        inputFunctions.forEach(Function::internalForward)

        if (hasValue().not()) {
            doForward(input.getRawValue(), weight.getRawValue(), bias?.getRawValue())
            nanCheck(output!!)
        }

        inputFunctions.forEach(Function::mayFreeOutput)
    }

    abstract fun doForward(input: AbstractRawTensor<Any>, weight: AbstractRawTensor<Any>, bias: AbstractRawTensor<Any>?)

}

internal class ConvND(
    stride: List<Int>,
    padding: List<Int>,
    dilation: List<Int>,
    private val transposed: Boolean,
    outputPadding: List<Int>,
    private val groups: Int,
    ops: TensorOperations<AbstractRawTensor<Any>>
): AbstractConvND(ops) {

    private val stride: MutableList<Int> = stride.toMutableList()
    private val padding: MutableList<Int> = padding.toMutableList()
    private val dilation: MutableList<Int> = dilation.toMutableList()
    private val outputPadding: MutableList<Int> = outputPadding.toMutableList()
    private var kernelShape = listOf<Int>()

    private var convolutionOperation: ConvolutionOperation<AbstractRawTensor<Any>>? = null

    init {
        if (padding.any { it < 0 })
            throw IllegalArgumentException("Negative padding is not supported")
        if (outputPadding.any { it < 0 })
            throw IllegalArgumentException("Negative output padding is not supported")
    }

    override fun calculateOutputShape(): List<Int> {
        val dims = input.dimensions
        val res = mutableListOf(input.shape[0])
        res += if (transposed) weight.shape[1] * groups else weight.shape[0]

        for (d in 2 until dims) {
            val kernel = dilation[d - 2] * (weight.shape[d] - 1) + 1
            if (transposed)
                res += (input.shape[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) + kernel + outputPadding[d - 2]
            else
                res += (input.shape[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1
        }

        return res
    }

    private fun view1dAs2d() {
        if (stride.size == 1) {
            stride.add(0, 1)
            padding.add(0, 0)
            dilation.add(0, 1)
            outputPadding.add(0, 0)
        }
    }

    private fun view4d(tensor: AbstractRawTensor<Any>): AbstractRawTensor<Any> {
        assert(tensor.dimensions == 3)
        return tensor.unsqueeze(2)
    }

    private fun view3d(tensor: AbstractRawTensor<Any>): AbstractRawTensor<Any> {
        assert(tensor.dimensions == 4)
        return tensor.squeeze(2)
    }

    override fun doForward(input: AbstractRawTensor<Any>, weight: AbstractRawTensor<Any>, bias: AbstractRawTensor<Any>?) {
        var inputNorm = input
        var weightNorm = weight

        if (input.dimensions == 3) {
            view1dAs2d()
            inputNorm = view4d(input)
            weightNorm = view4d(weight)
        }

        kernelShape = weightNorm.shape.drop(2)

        if (groups == 1) {
            output = computeOutput(inputNorm, weightNorm, bias)
        } else {
            val outputs = mutableListOf<AbstractRawTensor<Any>>()
            for (group in 0 until groups) {
                val inputInGroup = subtensor(inputNorm, 1, group)
                val weightInGroup = subtensor(weight, 0, group)
                val biasInGroup = if (bias != null) subtensor(bias, 0, group) else null
                outputs += computeOutput(inputInGroup, weightInGroup, biasInGroup)
            }
            output = ops.concat(1, outputs)
        }

        if (bias != null)
            saveForBackward(inputNorm, weightNorm, bias)
        else
            saveForBackward(inputNorm, weightNorm)

        if (input.dimensions == 3) {
            output = view3d(output!!)
        }
    }

    private fun subtensor(tensor: AbstractRawTensor<Any>, axis: Int, group: Int): AbstractRawTensor<Any> {
        val size = tensor.shape[axis] / groups
        return tensor.narrow(axis, size * group, size)
    }

    private fun assertParamSize(paramSize: Int, vararg params: List<Int>) = params.forEach {
        if (it.size != paramSize)
            throw IllegalArgumentException("Invalid convolution parameter, all parameters must have size == ${paramSize}, " +
                    "but got parameter: ${it}")
    }

    private fun findConvolutionOp(inputDimensions: Int): ConvolutionOperation<AbstractRawTensor<Any>> {
        if (convolutionOperation == null) {
            if (transposed) {
                if (inputDimensions == 4) {
                    assertParamSize(2, kernelShape, stride, padding, dilation, outputPadding)
                    convolutionOperation = ops.spatialConvolutionTranspose(kernelShape, stride, padding, dilation, outputPadding)
                } else if (inputDimensions == 5) {
                    assertParamSize(3, kernelShape, stride, padding, dilation, outputPadding)
                    convolutionOperation = ops.volumetricConvolutionTranspose(kernelShape, stride, padding, dilation, outputPadding)
                }
            } else {
                if (inputDimensions == 4) {
                    assertParamSize(2, kernelShape, stride, padding, dilation)
                    convolutionOperation = ops.spatialConvolution(kernelShape, stride, padding, dilation)
                } else if (inputDimensions == 5) {
                    assertParamSize(3, kernelShape, stride, padding, dilation)
                    convolutionOperation = ops.volumetricConvolution(kernelShape, stride, padding, dilation)
                }
            }
        }

        if (convolutionOperation == null)
            throw IllegalStateException("Unsupported convolution parameters")

        return convolutionOperation!!
    }

    private fun computeOutput(
        input: AbstractRawTensor<Any>,
        weight: AbstractRawTensor<Any>,
        bias: AbstractRawTensor<Any>?,
    ): AbstractRawTensor<Any> {
        return findConvolutionOp(input.dimensions).calculateOutput(input, weight, bias)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input, weight) = valuesSaved
        val bias = if (valuesSaved.size == 3) valuesSaved[2] else null

        val gradOut = if (gradient.dimensions == 3) view4d(gradient) else gradient

        var gradInput: AbstractRawTensor<Any>? = null
        var gradWeight: AbstractRawTensor<Any>? = null
        var gradBias: AbstractRawTensor<Any>? = null

        if (this.input.requiresGrad) {
            if (groups == 1) {
                gradInput = computeGradInput(input, gradOut, weight)
            } else {
                val gradInputs = mutableListOf<AbstractRawTensor<Any>>()
                for (group in 0 until groups) {
                    val inputInGroup = subtensor(input, 1, group)
                    val gradOutInGroup = subtensor(gradOut, 1, group)
                    val weightInGroup = subtensor(weight, 0, group)

                    gradInputs += computeGradInput(inputInGroup, gradOutInGroup, weightInGroup)
                }

                gradInput = ops.concat(1, gradInputs)
            }
        }

        if (this.weight.requiresGrad || this.bias?.requiresGrad == true) {
            if (groups == 1) {
                val gradParams = computeGradParams(input, gradOut, weight, bias)
                gradWeight = gradParams.first
                gradBias = gradParams.second
            } else {
                val gradParams = mutableListOf<Pair<AbstractRawTensor<Any>, AbstractRawTensor<Any>?>>()
                for (group in 0 until groups) {
                    val inputInGroup = subtensor(input, 1, group)
                    val gradOutInGroup = subtensor(gradOut, 1, group)
                    val weightInGroup = subtensor(weight, 0, group)
                    val biasInGroup = if (bias != null) subtensor(bias, 0, group) else null

                    gradParams += computeGradParams(inputInGroup, gradOutInGroup, weightInGroup, biasInGroup)
                }

                gradWeight = ops.concat(0, gradParams.map { it.first })

                if (this.bias?.requiresGrad == true)
                    gradBias = ops.concat(0, gradParams.map { it.second!! })
            }
        }

        if (gradient.dimensions == 3) {
            if (gradInput != null)
                gradInput = view3d(gradInput)
            if (gradWeight != null)
                gradWeight = view3d(gradWeight)
        }

        convolutionOperation?.cleanup()
        convolutionOperation = null //just to be safe

        return listOf(gradInput, gradWeight, gradBias)
    }

    private fun computeGradInput(
        input: AbstractRawTensor<Any>,
        gradOut: AbstractRawTensor<Any>,
        weight: AbstractRawTensor<Any>
    ): AbstractRawTensor<Any> {
        return findConvolutionOp(input.dimensions).calculateGradInput(input, gradOut, weight)
    }

    private fun computeGradParams(
        input: AbstractRawTensor<Any>,
        gradOut: AbstractRawTensor<Any>,
        weight: AbstractRawTensor<Any>,
        bias: AbstractRawTensor<Any>?
    ): Pair<AbstractRawTensor<Any>, AbstractRawTensor<Any>?> {
        val gradWeight = ops.zerosLike(weight)
        val gradBias = if (bias != null) ops.zerosLike(bias) else null

        findConvolutionOp(input.dimensions).accumulateGradParams(input, gradOut, gradWeight, gradBias)

        return gradWeight to gradBias
    }

}
