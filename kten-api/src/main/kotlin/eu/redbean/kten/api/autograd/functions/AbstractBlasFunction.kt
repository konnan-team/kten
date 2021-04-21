package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.autograd.applyUnpack
import eu.redbean.kten.api.autograd.map
import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.autograd.utils.checkBlasShape
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

abstract class AbstractBlasFunction(
    ops: TensorOperations<AbstractRawTensor<Any>>
): Function(ops) {

    protected lateinit var inputs: Triple<Tensor, Tensor, Tensor>
    protected var alpha = 1f
    protected var beta = 1f

    protected lateinit var addTensorShape: List<Int>

    operator fun invoke(addTensor: Tensor, tensor1: Tensor, tensor2: Tensor, alpha: Float = 1f, beta: Float = 1f): AbstractBlasFunction {
        addTensor.shape.checkBlasShape(tensor1.shape, tensor2.shape)
        cachedShape = addTensor.shape
        inputs = Triple(addTensor, tensor1, tensor2)
        this.alpha = alpha
        this.beta = beta
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
            addTensorShape = rawInputs.first.shape
            rawInputs.applyUnpack(this::doForward)
            nanCheck(output!!)
        }

        inputFunctions.forEach(Function::mayFreeOutput)
    }

    protected fun calculateAddTensorGrad(gradient: AbstractRawTensor<Any>): AbstractRawTensor<Any>? {
        var addTensorGrad: AbstractRawTensor<Any>? = null
        if (inputs.first is AGTensor && inputs.first.requiresGrad) {
            addTensorGrad = mayUnexpand(gradient, addTensorShape)
            if (beta != 1f)
                addTensorGrad = addTensorGrad * beta
        }
        return addTensorGrad
    }

    protected abstract fun doForward(addTensor: AbstractRawTensor<Any>, tensor1: AbstractRawTensor<Any>, tensor2: AbstractRawTensor<Any>)

}