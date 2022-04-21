package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

/**
 * Broadcasting to the common shape can mean unsqueeze or expand dimensions e.g.:
 * NxM op 1xNxM --> 1xNxM
 * or
 * 1xM op NxM --> NxM
 *
 * So this function takes care of both cases for the gradients.
 */
fun mayUnexpand(rawTensor: AbstractRawTensor<Any>, oldShape: List<Int>, ops: TensorOperations<AbstractRawTensor<Any>>): AbstractRawTensor<Any> {
    val unsqueezedNumberOfDims = rawTensor.dimensions - oldShape.size
    val expandedDims = rawTensor.shape.drop(unsqueezedNumberOfDims).zip(oldShape)
        .mapIndexed { index, (expanded, original) -> index to (expanded != original) }
        .filter { it.second }
        .map { it.first }

    var res = rawTensor

    var tensorToRelease: AbstractRawTensor<Any>
    for (i in 0 until unsqueezedNumberOfDims) {
        tensorToRelease = res
        res = res.sum(0, keepDimensions = false)
        if (i > 0) {
            ops.release(tensorToRelease)
        }
    }

    for ((i, it) in expandedDims.withIndex()) {
        tensorToRelease = res
        res = res.sum(it, keepDimensions = true)
        if (i > 0) {
            ops.release(tensorToRelease)
        }
    }

    return res
}

fun <T: Number, R: Number> Iterable<T>.accumulate(initial: R, operator: (acc: R, T) -> R): Sequence<R> = sequence {
    var last = initial
    forEach {
        last = operator(last, it)
        yield(last)
    }
}

