package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

abstract class UnaryTensorFunction(
    ops: TensorOperations<AbstractRawTensor<Any>>
): Function(ops) {

    protected lateinit var tensor: Tensor

    protected var modifiesShape = false

    operator fun invoke(tensor: Tensor): UnaryTensorFunction {
        if (!modifiesShape)
            cachedShape = tensor.shape.toList()
        this.tensor = tensor
        return this
    }

    override fun getInputsAsList(): List<Tensor> {
        return listOf(tensor)
    }

    override fun internalForward() {
        val inputFunction = if (tensor is Function) tensor as Function else null
        inputFunction?.internalForward()

        if (hasValue().not()) {
            doForward(tensor.getRawValue())
            nanCheck(output!!)
        }

        inputFunction?.mayFreeOutput()
    }

    protected abstract fun doForward(input: AbstractRawTensor<Any>)

}