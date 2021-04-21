package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.utils.toStoreSize
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

class Sum(
    ops: TensorOperations<AbstractRawTensor<Any>> //TODO add no axis variants for all of them
): AggregatorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        inputShape = input.shape
        if (axis == Int.MIN_VALUE)
            output = input.sum()
        else
            output = input.sum(axis, keepDimensions)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        if (!keepDimensions && inputShape.size != 1 && axis != Int.MIN_VALUE)
            gradient.inplaceUnsqueeze(axis)

        //TODO check if broadcast is good for all cases or do we need a repeat operation instead
        return listOf(gradient.broadcastTo(inputShape))
    }
}

class Mean(
    ops: TensorOperations<AbstractRawTensor<Any>>
): AggregatorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        inputShape = input.shape
        if (axis == Int.MIN_VALUE)
            output = input.mean()
        else
            output = input.mean(axis, keepDimensions)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        if (axis == Int.MIN_VALUE) {
            val gradInput = gradient / inputShape.toStoreSize().toFloat()
            return listOf(gradInput.broadcastTo(inputShape))
        }

        if (!keepDimensions && inputShape.size != 1)
            gradient.inplaceUnsqueeze(axis)

        //TODO check if broadcast is good for all cases or do we need a repeat operation instead
        val gradInput = gradient.broadcastTo(inputShape)
        gradInput /= inputShape[axis].toFloat()
        return listOf(gradInput)
    }
}

abstract class SelectionFunction(
    ops: TensorOperations<AbstractRawTensor<Any>>,
    val selection: ((AbstractRawTensor<Any>), (Int), (Boolean)) -> Pair<AbstractRawTensor<Any>, AbstractRawTensor<Any>>
): AggregatorFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        inputShape = input.shape
        val (res, indices) = selection(input, axis, keepDimensions)
        saveForBackward(input, indices)
        output = res
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val (input, indices) = valuesSaved
        val gradInput = ops.zerosLike(input) //TODO optimize out input saving by creating raw tensor on same platform by shape

        if (!keepDimensions && inputShape.size != 1) {
            gradient.inplaceUnsqueeze(axis)
            indices.inplaceUnsqueeze(axis)
        }

        gradInput.inplaceScatter(axis, indices, gradient)

        return listOf(gradInput)
    }
}

class Max(
    ops: TensorOperations<AbstractRawTensor<Any>>
): SelectionFunction(ops, { tensor, axis, keepDimensions ->
    tensor.max(axis, keepDimensions) to tensor.argMax(axis, keepDimensions) //TODO merge into one operation, and modify the calculation to have argmax/argmin
})

class Min(
    ops: TensorOperations<AbstractRawTensor<Any>>
): SelectionFunction(ops, { tensor, axis, keepDimensions ->
    tensor.min(axis, keepDimensions) to tensor.argMin(axis, keepDimensions)
})


