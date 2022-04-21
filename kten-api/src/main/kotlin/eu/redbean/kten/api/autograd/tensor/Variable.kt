package eu.redbean.kten.api.autograd.tensor

import eu.redbean.kten.api.autograd.functions.Function
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.serialization.CommonSerializableTensorDescriptor
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

/**
 * Note: if the raw value could be updated, then it wouldn't be necessary to rebuild the graph,
 * but in that case the flow control operations wouldn't take effect, so this option won't be
 * implemented yet.
 */
class Variable(
    ops: TensorOperations<AbstractRawTensor<Any>>,
    private val value: AbstractRawTensor<Any>,
    private val gradient: AbstractRawTensor<Any> = ops.zerosLike(value)
): AGTensor(ops) {

    override val shape: List<Int>
        get() = value.shape

    override fun forward() {
        //no-op
    }

    override fun backward() {
        //no-op
    }

    override fun backward(gradients: Tensor) {
        if (shape != gradients.shape)
            throw IllegalArgumentException("Variable with shape: $shape cannot have gradients with shape: ${gradients.shape}")

        if (gradients is Function && gradients.hasValue().not())
            gradients.forward()

        backwardWithGrad(gradients.getRawValue())
    }

    override fun getRawValue(): AbstractRawTensor<Any> = value

    override fun backwardWithGrad(gradient: AbstractRawTensor<Any>) {
        this.gradient += gradient
    }

    fun zeroGrad() {
        ops.zeroOut(gradient)
    }

    override fun grad(): Tensor = NoGradVariable(ops, gradient)

    fun inplaceMultiplyGrad(value: Float) {
        gradient *= value
    }

    override fun retainGrad(): Tensor {
        return this
    }

    override fun asVariable(requiresGrad: Boolean): Tensor {
        if (requiresGrad)
            return this

        return super.asVariable(requiresGrad)
    }

    fun inplaceAddToValue(tensor: Tensor) {
        if (tensor !is NoGradVariable)
            throw IllegalArgumentException("Inplace raw update is only allowed with no-grad variables")

        value += tensor.getRawValue()
    }

    fun inplaceSetValue(tensor: Tensor) {
        if (this.shape != tensor.shape)
            throw IllegalArgumentException("Invalid value shape")

        ops.zeroOut(value)
        value += tensor.getRawValue()
    }

    override fun toPlatform(platform: String): Tensor {
        if (ops.platformKey == platform)
            return this

        return Variable(
            PlatformProvider.platformOps(platform),
            PlatformProvider.transformRawData(value, ops.platformKey, platform),
            PlatformProvider.transformRawData(gradient, ops.platformKey, platform)
        )
    }

    override fun release() {
        ops.release(value, gradient)
    }

    override fun incrementRef() {
        ops.incrementRef(value)
        ops.incrementRef(gradient)
    }

    override fun serialize(): CommonSerializableTensorDescriptor {
        return CommonSerializableTensorDescriptor(
            ops.toSerializableData(value),
            ops.toSerializableData(gradient)
        )
    }

}