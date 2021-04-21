package eu.redbean.kten.api.tensor.operations

import eu.redbean.kten.api.tensor.Tensor
import java.io.Closeable

fun interface TensorOperationsGarbageCollector: Closeable {

    fun mustKeep(vararg tensors: Tensor) {
        tensors.forEach { it.incrementRef() }
    }

}