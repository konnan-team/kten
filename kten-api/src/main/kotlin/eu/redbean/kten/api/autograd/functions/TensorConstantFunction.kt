package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

abstract class TensorConstantFunction(
    ops: TensorOperations<AbstractRawTensor<Any>>
): UnaryTensorFunction(ops) {

    protected var constant: Float = 0.0f
    protected var tensorFirst = true

    operator fun invoke(tensor: Tensor, constant: Float): TensorConstantFunction {
        invoke(tensor)
        this.constant = constant
        return this
    }

    operator fun invoke(constant: Float, tensor: Tensor): TensorConstantFunction {
        this.tensorFirst = false
        return invoke(tensor, constant)
    }

}