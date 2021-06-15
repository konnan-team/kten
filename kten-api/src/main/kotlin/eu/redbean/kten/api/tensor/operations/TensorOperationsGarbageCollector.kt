package eu.redbean.kten.api.tensor.operations

import eu.redbean.kten.api.autograd.functions.Function
import eu.redbean.kten.api.autograd.tensor.NoGradVariable
import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.Tensor
import java.io.Closeable

fun interface TensorOperationsGarbageCollector: Closeable {

    fun mustKeep(vararg tensors: Tensor) {
        if (tensors.any { it is Function }) {
            throw IllegalArgumentException("Only explicit Variables are allowed to keep between GC calls")
        }
        tensors.forEach {
            it.platformOps().markSurviveGC(it.getRawValue())
            if (it.requiresGrad)
                it.platformOps().markSurviveGC(it.grad().getRawValue())
        }
    }

    fun mayRelease(vararg tensors: Tensor) {
        if (tensors.any { it is Function }) {
            throw IllegalArgumentException("Only explicit Variables are allowed to be marked as releasable in GC")
        }

        tensors.forEach {
            it.platformOps().markReleasableInGC(it.getRawValue())
            if (it.requiresGrad)
                it.platformOps().markReleasableInGC(it.grad().getRawValue())
        }
    }

}