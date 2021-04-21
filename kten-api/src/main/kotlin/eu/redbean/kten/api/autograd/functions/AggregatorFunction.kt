package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.utils.aggregateOver
import eu.redbean.kten.api.autograd.utils.normalizeAxis
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

abstract class AggregatorFunction(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    protected var axis = Int.MIN_VALUE
    protected var keepDimensions = false
    protected lateinit var inputShape: List<Int>

    init {
        modifiesShape = true
    }

    operator fun invoke(tensor: Tensor, axis: Int, keepDimensions: Boolean): AggregatorFunction {
        if (axis == Int.MIN_VALUE)
            cachedShape = listOf(1)
        else
            cachedShape = tensor.shape.aggregateOver(axis, keepDimensions)
        invoke(tensor)
        this.axis = if (axis == Int.MIN_VALUE) axis else tensor.shape.normalizeAxis(axis)
        this.keepDimensions = keepDimensions
        return this
    }

}