package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.tensor.store.AbstractRawTensor

/**
 * Broadcasting to the common shape can mean unsqueeze or expand dimensions e.g.:
 * NxM op 1xNxM --> 1xNxM
 * or
 * 1xM op NxM --> NxM
 *
 * So this function takes care of both cases for the gradients.
 */
fun mayUnexpand(rawTensor: AbstractRawTensor<Any>, oldShape: List<Int>): AbstractRawTensor<Any> {
    val unsqueezedNumberOfDims = rawTensor.dimensions - oldShape.size
    val expandedDims = rawTensor.shape.drop(unsqueezedNumberOfDims).zip(oldShape)
        .mapIndexed { index, (expanded, original) -> index to (expanded != original) }
        .filter { it.second }
        .map { it.first }

    var res = rawTensor

    for (i in 0 until unsqueezedNumberOfDims)
        res = res.sum(0, keepDimensions = false)

    expandedDims.forEach {
        res = res.sum(it, keepDimensions = true)
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

