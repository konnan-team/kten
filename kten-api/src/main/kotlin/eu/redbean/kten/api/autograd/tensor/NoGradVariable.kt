package eu.redbean.kten.api.autograd.tensor

import eu.redbean.kten.api.autograd.utils.checkBlasShape
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.serialization.CommonSerializableTensorDescriptor
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import kotlin.IllegalStateException

class NoGradVariable(
    ops: TensorOperations<AbstractRawTensor<Any>>,
    private val rawValue: AbstractRawTensor<Any>
): AGTensor(ops) {

    override val requiresGrad: Boolean
        get() = false

    override val shape: List<Int>
        get() = rawValue.shape

    override fun backwardWithGrad(gradient: AbstractRawTensor<Any>) {
        throw IllegalStateException("Tensor does not require gradients")
    }

    override fun forward() {
        // no-op
    }

    override fun backward() {
        throw IllegalStateException("Tensor does not require gradients")
    }

    override fun backward(gradients: Tensor) {
        throw IllegalStateException("Tensor does not require gradients")
    }

    override fun getRawValue(): AbstractRawTensor<Any> = rawValue

    private fun createNew(raw: AbstractRawTensor<Any>): NoGradVariable = NoGradVariable(ops, raw)

    override fun get(vararg index: Int): Tensor = createNew(rawValue[index])

    override fun get(vararg index: IntRange): Tensor = createNew(rawValue[index])

    override fun set(vararg index: Int, value: Tensor) {
        if (value.requiresGrad)
            value.forward()
        this.rawValue[index] = value.getRawValue()
    }

    override fun set(vararg index: IntRange, value: Tensor) {
        if (value.requiresGrad)
            value.forward()
        this.rawValue[index] = value.getRawValue()
    }

    override fun set(vararg index: Int, value: Float) {
        this.rawValue[index] = value
    }

    override fun set(vararg index: IntRange, value: Float) {
        this.rawValue[index] = value
    }

    override fun plus(other: Tensor): Tensor {
        if (other.requiresGrad)
            return super.plus(other)
        return createNew(this.rawValue + other.getRawValue())
    }

    override fun plus(constant: Float): Tensor = createNew(this.rawValue + constant)

    override fun Float.plus(tensor: Tensor): Tensor = createNew(tensor.getRawValue() + this)

    override fun minus(other: Tensor): Tensor {
        if (other.requiresGrad)
            return super.minus(other)
        return createNew(this.rawValue - other.getRawValue())
    }

    override fun minus(constant: Float): Tensor = createNew(this.rawValue - constant)

    override fun Float.minus(tensor: Tensor): Tensor = createNew(tensor.getRawValue().unaryMinus() + this)

    override fun times(other: Tensor): Tensor {
        if (other.requiresGrad)
            return super.times(other)
        return createNew(this.rawValue * other.getRawValue())
    }

    override fun times(constant: Float): Tensor = createNew(this.rawValue * constant)

    override fun Float.times(tensor: Tensor): Tensor = createNew(tensor.getRawValue() * this)

    override fun div(other: Tensor): Tensor {
        if (other.requiresGrad)
            return super.div(other)
        return createNew(this.rawValue / other.getRawValue())
    }

    override fun div(constant: Float): Tensor = createNew(this.rawValue / constant)

    override fun Float.div(tensor: Tensor): Tensor = createNew(tensor.getRawValue().reciprocal() * this)

    override fun pow(other: Tensor): Tensor {
        if (other.requiresGrad)
            return super.pow(other)
        return createNew(this.rawValue.pow(other.getRawValue()))
    }

    override fun pow(constant: Float): Tensor = createNew(this.rawValue.pow(constant))

    override fun Float.pow(tensor: Tensor): Tensor = createNew(ops.pow(this, tensor.getRawValue()))

    override fun sum(axis: Int, keepDimensions: Boolean): Tensor = createNew(this.rawValue.sum(axis, keepDimensions))

    override fun mean(axis: Int, keepDimensions: Boolean): Tensor = createNew(this.rawValue.mean(axis, keepDimensions))

    override fun max(axis: Int, keepDimensions: Boolean): Tensor = createNew(this.rawValue.max(axis, keepDimensions))

    override fun min(axis: Int, keepDimensions: Boolean): Tensor = createNew(this.rawValue.min(axis, keepDimensions))

    override fun sum(): Tensor = createNew(this.rawValue.sum())

    override fun mean(): Tensor = createNew(this.rawValue.mean())

    override fun exp(): Tensor = createNew(this.rawValue.exp())

    override fun log(): Tensor = createNew(this.rawValue.log())

    override fun tanh(): Tensor = createNew(this.rawValue.tanh())

    override fun sigmoid(): Tensor = createNew(this.rawValue.sigmoid())

    override fun sinh(): Tensor = createNew(this.rawValue.sinh())

    override fun cosh(): Tensor = createNew(this.rawValue.cosh())

    override fun abs(): Tensor = createNew(this.rawValue.abs())

    override fun clamp(min: Float, max: Float): Tensor = createNew(this.rawValue.clamp(min, max))

    override fun sqrt(): Tensor = createNew(this.rawValue.sqrt())

    override fun sin(): Tensor = createNew(this.rawValue.sin())

    override fun cos(): Tensor = createNew(this.rawValue.cos())

    override fun tan(): Tensor = createNew(this.rawValue.tan())

    override fun asin(): Tensor = createNew(this.rawValue.asin())

    override fun acos(): Tensor = createNew(this.rawValue.acos())

    override fun atan(): Tensor = createNew(this.rawValue.atan())

    override fun reciprocal(): Tensor = createNew(this.rawValue.reciprocal())

    override fun floor(): Tensor = createNew(this.rawValue.floor())

    override fun ceil(): Tensor = createNew(this.rawValue.ceil())

    override fun round(): Tensor = createNew(this.rawValue.round())

    override fun sign(): Tensor = createNew(this.rawValue.sign())

    override fun trunc(): Tensor = createNew(this.rawValue.trunc())

    override fun rsqrt(): Tensor = createNew(this.rawValue.rsqrt())

    override fun transpose(axis1: Int, axis2: Int): Tensor = createNew(this.rawValue.transpose(axis1, axis2))

    override fun reshape(newShape: List<Int>): Tensor = createNew(this.rawValue.reshape(newShape))

    override fun view(newShape: List<Int>): Tensor = createNew(this.rawValue.view(newShape))

    override fun expand(newShape: List<Int>): Tensor = createNew(this.rawValue.broadcastTo(newShape))

    override fun permute(axes: List<Int>): Tensor = createNew(this.rawValue.permute(axes))

    override fun concat(axis: Int, tensors: List<Tensor>): Tensor {
        if (tensors.any { it.requiresGrad })
            return super.concat(axis, tensors)
        return createNew(ops.concat(axis, tensors.map { it.getRawValue() }))
    }

    override fun copy(): Tensor = createNew(this.rawValue.copy())

    override fun squeeze(axis: Int): Tensor = createNew(this.rawValue.squeeze(axis))

    override fun unsqueeze(axis: Int): Tensor = createNew(this.rawValue.unsqueeze(axis))

    override fun gather(axis: Int, index: Tensor): Tensor = createNew(this.rawValue.gather(axis, index.getRawValue()))

    override fun scatter(axis: Int, index: Tensor, source: Tensor): Tensor {
        if (source.requiresGrad)
            return super.scatter(axis, index, source)
        return createNew(this.rawValue.scatter(axis, index.getRawValue(), source.getRawValue()))
    }

    override fun indexSelect(axis: Int, index: Tensor): Tensor = createNew(this.rawValue.indexSelect(axis, index.getRawValue()))

    override fun gemm(addMatrix: Tensor, matrix: Tensor, alpha: Float, beta: Float): Tensor {
        if (addMatrix.requiresGrad || matrix.requiresGrad)
            return super.gemm(addMatrix, matrix, alpha, beta)

        addMatrix.shape.checkBlasShape(this.shape, matrix.shape)
        return createNew(ops.gemm(addMatrix.getRawValue(), this.rawValue, matrix.getRawValue(), alpha, beta))
    }

    override fun batchedGemm(addTensor: Tensor, tensor: Tensor, alpha: Float, beta: Float): Tensor {
        if (addTensor.requiresGrad || tensor.requiresGrad)
            return super.batchedGemm(addTensor, tensor, alpha, beta)

        addTensor.shape.checkBlasShape(this.shape, tensor.shape)
        return createNew(ops.gemmBatched(addTensor.getRawValue(), this.rawValue, tensor.getRawValue(), alpha, beta))
    }

    override fun gemv(addVector: Tensor, vector: Tensor, alpha: Float, beta: Float): Tensor {
        if (addVector.requiresGrad || vector.requiresGrad)
            return super.gemv(addVector, vector, alpha, beta)

        addVector.shape.checkBlasShape(this.shape, vector.shape)
        return createNew(ops.gemv(addVector.getRawValue(), this.rawValue, vector.getRawValue(), alpha, beta))
    }

    override fun dot(vector: Tensor): Tensor {
        if (vector.requiresGrad)
            return super.dot(vector)
        return createNew(this.rawValue.dot(vector.getRawValue()))
    }

    override fun lt(tensor: Tensor): Tensor {
        if (tensor.requiresGrad)
            return super.lt(tensor)
        return createNew(this.rawValue.lt(tensor.getRawValue()))
    }

    override fun lte(tensor: Tensor): Tensor {
        if (tensor.requiresGrad)
            return super.lte(tensor)
        return createNew(this.rawValue.lte(tensor.getRawValue()))
    }

    override fun gt(tensor: Tensor): Tensor {
        if (tensor.requiresGrad)
            return super.gt(tensor)
        return createNew(this.rawValue.gt(tensor.getRawValue()))
    }

    override fun gte(tensor: Tensor): Tensor {
        if (tensor.requiresGrad)
            return super.gte(tensor)
        return createNew(this.rawValue.gte(tensor.getRawValue()))
    }

    override fun eq(tensor: Tensor): Tensor {
        if (tensor.requiresGrad)
            return super.eq(tensor)
        return createNew(this.rawValue.eq(tensor.getRawValue()))
    }

    override fun neq(tensor: Tensor): Tensor {
        if (tensor.requiresGrad)
            return super.neq(tensor)
        return createNew(this.rawValue.neq(tensor.getRawValue()))
    }

    override fun lt(value: Float): Tensor = createNew(this.rawValue.lt(value))

    override fun lte(value: Float): Tensor = createNew(this.rawValue.lte(value))

    override fun gt(value: Float): Tensor = createNew(this.rawValue.gt(value))

    override fun gte(value: Float): Tensor = createNew(this.rawValue.gte(value))

    override fun eq(value: Float): Tensor = createNew(this.rawValue.eq(value))

    override fun neq(value: Float): Tensor = createNew(this.rawValue.neq(value))

    override fun maskedFill(mask: Tensor, value: Float): Tensor = createNew(this.rawValue.maskedFill(mask.getRawValue(), value))

    override fun noGrad(): Tensor = this

    override fun grad(): Tensor {
        throw IllegalStateException("Tensor has no gradient")
    }

    override fun retainGrad(): Tensor {
        return this
    }

    override fun toPlatform(platform: String): Tensor {
        if (ops.platformKey == platform)
            return this

        return NoGradVariable(
            PlatformProvider.platformOps(platform),
            PlatformProvider.transformRawData(rawValue, ops.platformKey, platform)
        )
    }

    override fun release() {
        ops.release(rawValue)
    }

    override fun incrementRef() {
        ops.incrementRef(rawValue)
    }

    override fun serialize(): CommonSerializableTensorDescriptor {
        return CommonSerializableTensorDescriptor(
            ops.toSerializableData(rawValue),
            null
        )
    }

}