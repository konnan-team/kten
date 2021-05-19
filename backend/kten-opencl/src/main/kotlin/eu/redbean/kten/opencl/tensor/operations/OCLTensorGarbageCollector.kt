package eu.redbean.kten.opencl.tensor.operations

import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperationsGarbageCollector


fun interface OCLTensorGarbageCollector: TensorOperationsGarbageCollector {

    override fun mustKeep(vararg tensors: Tensor) {
        for (i in 0 until 1000) //TODO figure out something, because this won't work obviously
            super.mustKeep(*tensors)
    }

}