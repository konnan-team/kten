package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

abstract class AnyTensorFunction(
    ops: TensorOperations<AbstractRawTensor<Any>>
): Function(ops) {

    protected lateinit var inputTensors: List<Tensor>

    override fun getInputsAsList(): List<Tensor> {
        return inputTensors
    }

    override fun internalForward() {
        val inputFunctions = inputTensors.filter { it is Function }.map { it as Function }
        inputFunctions.forEach(Function::internalForward)

        if (hasValue().not()) {
            val rawInputs = inputTensors.map(Tensor::getRawValue)
            doForward(rawInputs)
            nanCheck(output!!)
        }

        inputFunctions.forEach(Function::mayFreeOutput)
    }

    protected abstract fun doForward(inputs: List<AbstractRawTensor<Any>>)

}