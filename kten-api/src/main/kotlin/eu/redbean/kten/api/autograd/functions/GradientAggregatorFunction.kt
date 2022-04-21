package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

class GradientAggregatorFunction(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    private lateinit var aggregatorVariable: Tensor
    private var calculation: (Tensor) -> Tensor = { it }
    private var result: Tensor? = null

    operator fun invoke(tensor: Tensor, calculation: (Tensor) -> Tensor, shape: List<Int>? = null): GradientAggregatorFunction {
        if (shape != null) {
            modifiesShape = true
            cachedShape = shape
        }
        super.invoke(tensor)
        this.calculation = calculation
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        if (tensor is Function) {
            aggregatorVariable = tensor.asVariable(requiresGrad = true)
            result = calculation(aggregatorVariable)
        } else {
            result = calculation(tensor)
        }
        result!!.forward()
        output = result!!.getRawValue()
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        if (tensor is Function) {
            val resultNonNull = result!!
            if (resultNonNull is AGTensor) {
                resultNonNull.backwardWithGrad(gradient)
            }
            val gradIn = aggregatorVariable.grad().getRawValue().copy()
            (aggregatorVariable as Variable).zeroGrad()
            return listOf(gradIn)
        } else {
            return listOf(ops.zerosLike(gradient))
        }
    }

    override fun releaseUnusedInGraph() {
        val resultNonNull = result!!
        if (resultNonNull is Function)
            resultNonNull.releaseUnusedInGraph()
        super.releaseUnusedInGraph()
    }
}