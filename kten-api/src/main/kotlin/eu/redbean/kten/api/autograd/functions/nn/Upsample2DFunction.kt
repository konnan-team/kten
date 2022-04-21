package eu.redbean.kten.api.autograd.functions.nn

import eu.redbean.kten.api.autograd.functions.UnaryTensorFunction
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.operations.nn.UpsampleType
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

class Upsample2DFunction(
    upsampleType: UpsampleType,
    val scale: Int,
    ops: TensorOperations<AbstractRawTensor<Any>>
) : UnaryTensorFunction(ops) {

    private val upsampleOperation = ops.upsample(upsampleType, scale)

    private lateinit var inputShape: List<Int>

    override operator fun invoke(tensor: Tensor): Upsample2DFunction {
        modifiesShape = true
        inputShape = tensor.shape
        upsampleOperation.checkDimensions(inputShape)
        cachedShape = upsampleOperation.calculateOutputShape(inputShape)
        super.invoke(tensor)
        return this
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        output = upsampleOperation.upsample(input)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(upsampleOperation.calculateGrad(gradient, inputShape))
    }
}