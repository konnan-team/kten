package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.applyUnpack
import eu.redbean.kten.api.autograd.map
import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.autograd.utils.inferImplicitBroadcastShape
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

abstract class BiTensorFunction(
    ops: TensorOperations<AbstractRawTensor<Any>>,
    val broadcastingOperation: Boolean = false
): Function(ops) {

    protected lateinit var inputs: Pair<Tensor, Tensor>

    protected lateinit var aShape: List<Int>
    protected lateinit var bShape: List<Int>

    open operator fun invoke(a: Tensor, b: Tensor): BiTensorFunction {
        if (broadcastingOperation)
            cachedShape = inferImplicitBroadcastShape(a.shape, b.shape)
        this.inputs = a to b
        return this
    }

    override fun getInputsAsList(): List<Tensor> {
        return inputs.toList()
    }

    override fun internalForward() {
        val inputFunctions = inputs.toList().filter { it is Function }.map { it as Function }
        inputFunctions.forEach(Function::internalForward)

        if (hasValue().not()) {
            val rawInputs = inputs.map(Tensor::getRawValue)
            aShape = rawInputs.first.shape
            bShape = rawInputs.second.shape
            rawInputs.applyUnpack(this::doForward)
            nanCheck(output!!)
        }

        inputFunctions.forEach(Function::mayFreeOutput)
    }

    protected abstract fun doForward(a: AbstractRawTensor<Any>, b: AbstractRawTensor<Any>)

}