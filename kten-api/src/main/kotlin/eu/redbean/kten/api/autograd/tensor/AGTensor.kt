package eu.redbean.kten.api.autograd.tensor

import eu.redbean.kten.api.autograd.functions.*
import eu.redbean.kten.api.autograd.functions.Function
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import eu.redbean.kten.api.autograd.utils.*
import eu.redbean.kten.api.tensor.Constants.end
import eu.redbean.kten.api.tensor.Constants.start
import eu.redbean.kten.api.tensor.serialization.CommonSerializableTensorDescriptor

abstract class AGTensor(
    ops: TensorOperations<AbstractRawTensor<Any>>
) : Tensor(ops) {

    override val requiresGrad: Boolean
        get() = true

    override val platform: String
        get() = ops.platformKey


    private var inplaceSafeThis: AGTensor? = null

    private fun inplaceSafe(tensor: Tensor): Tensor {
        if (tensor is AGTensor)
            return tensor.inplaceSafe()
        return tensor
    }

    fun inplaceSafe(): AGTensor {
        return inplaceSafeThis ?: this
    }

    override fun item(): Float {
        val safeThis = inplaceSafeForwarded()

        if (safeThis.dimensions != 1 || safeThis.shape[0] != 1)
            throw IllegalStateException(
                "Item can only be accessed from singleton tensors (1D tensors with 1 element in them), " +
                        "but tensor shape is ${safeThis.shape}"
            )

        return safeThis.getRawValue().getSingleValue(listOf(0))
    }

    override fun getValue(index: List<Int>): Float {
        return inplaceSafeForwarded().getRawValue().getSingleValue(index)
    }

    private fun inplaceSafeForwarded(): Tensor {
        val safeThis = inplaceSafe(this)

        if (safeThis is Function && safeThis.hasValue().not())
            safeThis.forward()

        return safeThis
    }

    override fun get(vararg index: Int): Tensor {
        return Index(ops)(inplaceSafe(this), index)
    }

    override fun get(vararg index: IntRange): Tensor {
        return Index(ops)(inplaceSafe(this), index)
    }

    override fun set(vararg index: Int, value: Tensor) {
        inplaceSafeThis = IndexSetTensor(ops)(inplaceSafe(this), index, inplaceSafe(value))
    }

    override fun set(vararg index: IntRange, value: Tensor) {
        inplaceSafeThis = IndexSetTensor(ops)(inplaceSafe(this), index, inplaceSafe(value))
    }

    override fun set(vararg index: Int, value: Float) {
        inplaceSafeThis = IndexSetConstant(ops)(inplaceSafe(this), index, value)
    }

    override fun set(vararg index: IntRange, value: Float) {
        inplaceSafeThis = IndexSetConstant(ops)(inplaceSafe(this), index, value)
    }

    override fun plus(other: Tensor): Tensor {
        return Plus(ops)(inplaceSafe(this), inplaceSafe(other))
    }

    override fun plus(constant: Float): Tensor {
        return PlusConstant(ops)(inplaceSafe(this), constant)
    }

    override fun Float.plus(tensor: Tensor): Tensor {
        return PlusConstant(ops)(this, inplaceSafe(tensor))
    }

    override fun minus(other: Tensor): Tensor {
        return Minus(ops)(inplaceSafe(this), inplaceSafe(other))
    }

    override fun minus(constant: Float): Tensor {
        return MinusConstant(ops)(inplaceSafe(this), constant)
    }

    override fun Float.minus(tensor: Tensor): Tensor {
        return MinusConstant(ops)(this, inplaceSafe(tensor))
    }

    override fun times(other: Tensor): Tensor {
        return Times(ops)(inplaceSafe(this), inplaceSafe(other))
    }

    override fun times(constant: Float): Tensor {
        return TimesConstant(ops)(inplaceSafe(this), constant)
    }

    override fun Float.times(tensor: Tensor): Tensor {
        return TimesConstant(ops)(this, inplaceSafe(tensor))
    }

    override fun div(other: Tensor): Tensor {
        return Div(ops)(inplaceSafe(this), inplaceSafe(other))
    }

    override fun div(constant: Float): Tensor {
        return DivConstant(ops)(inplaceSafe(this), constant)
    }

    override fun Float.div(tensor: Tensor): Tensor {
        return DivConstant(ops)(this, inplaceSafe(tensor))
    }

    override fun pow(other: Tensor): Tensor {
        return Pow(ops)(inplaceSafe(this), inplaceSafe(other))
    }

    override fun pow(constant: Float): Tensor {
        return PowConstant(ops)(inplaceSafe(this), constant)
    }

    override fun Float.pow(tensor: Tensor): Tensor {
        return PowConstant(ops)(this, inplaceSafe(tensor))
    }

    override fun sum(axis: Int, keepDimensions: Boolean): Tensor {
        return Sum(ops)(inplaceSafe(this), axis, keepDimensions)
    }

    override fun mean(axis: Int, keepDimensions: Boolean): Tensor {
        return Mean(ops)(inplaceSafe(this), axis, keepDimensions)
    }

    override fun max(axis: Int, keepDimensions: Boolean): Tensor {
        return Max(ops)(inplaceSafe(this), axis, keepDimensions)
    }

    override fun min(axis: Int, keepDimensions: Boolean): Tensor {
        return Min(ops)(inplaceSafe(this), axis, keepDimensions)
    }

    override fun sum(): Tensor {
        return Sum(ops)(inplaceSafe(this), axis = Int.MIN_VALUE, keepDimensions = false)
    }

    override fun mean(): Tensor {
        return Mean(ops)(inplaceSafe(this), axis = Int.MIN_VALUE, keepDimensions = false)
    }

    override fun argMax(axis: Int, keepDimensions: Boolean): Tensor {
        return NoGradVariable(ops, inplaceSafeForwarded().getRawValue().argMax(axis, keepDimensions))
    }

    override fun argMin(axis: Int, keepDimensions: Boolean): Tensor {
        return NoGradVariable(ops, inplaceSafeForwarded().getRawValue().argMin(axis, keepDimensions))
    }

    override fun exp(): Tensor {
        return Exp(ops)(inplaceSafe(this))
    }

    override fun log(): Tensor {
        return Log(ops)(inplaceSafe(this))
    }

    override fun tanh(): Tensor {
        return Tanh(ops)(inplaceSafe(this))
    }

    override fun sigmoid(): Tensor {
        return Sigmoid(ops)(inplaceSafe(this))
    }

    override fun sinh(): Tensor {
        return Sinh(ops)(inplaceSafe(this))
    }

    override fun cosh(): Tensor {
        return Cosh(ops)(inplaceSafe(this))
    }

    override fun abs(): Tensor {
        return Abs(ops)(inplaceSafe(this))
    }

    override fun clamp(min: Float, max: Float): Tensor {
        return Clamp(ops)(inplaceSafe(this), min, max)
    }

    override fun sqrt(): Tensor {
        return Sqrt(ops)(inplaceSafe(this))
    }

    override fun sin(): Tensor {
        return Sin(ops)(inplaceSafe(this))
    }

    override fun cos(): Tensor {
        return Cos(ops)(inplaceSafe(this))
    }

    override fun tan(): Tensor {
        return Tan(ops)(inplaceSafe(this))
    }

    override fun asin(): Tensor {
        return Asin(ops)(inplaceSafe(this))
    }

    override fun acos(): Tensor {
        return Acos(ops)(inplaceSafe(this))
    }

    override fun atan(): Tensor {
        return Atan(ops)(inplaceSafe(this))
    }

    override fun reciprocal(): Tensor {
        return Reciprocal(ops)(inplaceSafe(this))
    }

    override fun floor(): Tensor {
        return Floor(ops)(inplaceSafe(this))
    }

    override fun ceil(): Tensor {
        return Ceil(ops)(inplaceSafe(this))
    }

    override fun round(): Tensor {
        return Round(ops)(inplaceSafe(this))
    }

    override fun sign(): Tensor {
        return Sign(ops)(inplaceSafe(this))
    }

    override fun trunc(): Tensor {
        return Trunc(ops)(inplaceSafe(this))
    }

    override fun rsqrt(): Tensor {
        return Rsqrt(ops)(inplaceSafe(this))
    }

    override fun transpose(axis1: Int, axis2: Int): Tensor {
        return Transpose(ops)(inplaceSafe(this), axis1, axis2)
    }

    override fun reshape(newShape: List<Int>): Tensor {
        return Reshape(ops)(inplaceSafe(this), newShape)
    }

    override fun expand(newShape: List<Int>): Tensor {
        return Expand(ops)(inplaceSafe(this), newShape)
    }

    override fun permute(axes: List<Int>): Tensor {
        return Permute(ops)(inplaceSafe(this), axes)
    }

    override fun concat(axis: Int, tensors: List<Tensor>): Tensor {
        return Concat(ops)(axis, tensors.map(this::inplaceSafe))
    }

    override fun copy(): Tensor {
        return Copy(ops)(inplaceSafe(this))
    }

    override fun squeeze(axis: Int): Tensor {
        return Squeeze(ops)(inplaceSafe(this), axis)
    }

    override fun unsqueeze(axis: Int): Tensor {
        return Unsqueeze(ops)(inplaceSafe(this), axis)
    }

    override fun gather(axis: Int, index: Tensor): Tensor {
        return Gather(ops)(inplaceSafe(this), axis, index) //No need to inplace guard index, because it cannot be differentiated anyway
    }

    override fun scatter(axis: Int, index: Tensor, source: Tensor): Tensor {
        return Scatter(ops)(inplaceSafe(this), axis, index, inplaceSafe(source))
    }

    override fun indexSelect(axis: Int, index: Tensor): Tensor {
        return IndexSelect(ops)(inplaceSafe(this), index, axis)
    }

    override fun matmul(tensor: Tensor): Tensor {
        val t1 = inplaceSafe(this)
        val t2 = inplaceSafe(tensor)

        t1.shape.checkMatmulShapesCompatible(t2.shape)

        val dim1 = t1.dimensions
        val dim2 = t2.dimensions

        if (dim1 == 1 && dim2 == 1)
            return t1.dot(t2)
        else if (dim1 == 2 && dim2 == 1)
            return t1.mv(t2)
        else if (dim1 == 1 && dim2 == 2)
            return t1.unsqueeze(0).mm(t2).squeeze(0)
        else if (dim1 == 2 && dim2 == 2)
            return t1.mm(t2)
        else if (dim1 >= 3 && (dim2 < 3)) {
            return optimizationSpecificBatchMatMul(t1, t2, dim2)
        } else if ((dim1 >= 1 && dim2 >= 1) && (dim1 >= 3 || dim2 >= 3)) {
            return broadcastingBatchMatMul(t1, dim1, t2, dim2)
        }

        throw IllegalArgumentException("Matmul can only be used on tensors with at least one dimensions, but got tensors with dimensions: $dim1 and $dim2")
    }

    private fun optimizationSpecificBatchMatMul(tensor1: Tensor, tensor2: Tensor, dim2: Int): Tensor {
        var t1 = tensor1
        var t2 = tensor2
        if (dim2 == 1)
            t2 = t2.unsqueeze(-1)

        val shape1 = t1.shape
        val shape2 = t2.shape
        val outShape = shape1[start..-1] + shape2[-1..end]

        //folding batches to first dimension to enable use of mm instead of bmm
        t1 = t1.view(-1, shape1.last())

        var res = t1.mm(t2).view(outShape)

        if (dim2 == 1)
            res = res.squeeze(-1)

        return res
    }

    private fun broadcastingBatchMatMul(tensor1: Tensor, dim1: Int, tensor2: Tensor, dim2: Int): Tensor {
        // Ensures the shape at least 3 dimensions by adding singleton dimensions to the beginning
        var t2 = tensor2
        val t1ExpandShape = listOf(1) * kotlin.math.max(3 - tensor1.dimensions, 0) + tensor1.shape

        // If rhs is a vector we create a matrix from it by unsqueezing the last dimension, then we can add the batch dimension
        if (dim2 == 1)
            t2 = t2.unsqueeze(1)
        val t2ExpandShape = listOf(1) * kotlin.math.max(3 - t2.dimensions, 0) + t2.shape

        val expandBatchPortion = inferImplicitBroadcastShape(t1ExpandShape[start..-2], t2ExpandShape[start..-2])

        val t1MatrixShape = t1ExpandShape[-2..end]
        val t2MatrixShape = t2ExpandShape[-2..end]

        // broadcasting to the new shape with the batch dimensions (to ensure the singleton dimensions are repeated correctly)
        // than reshape to the single batch dimension shape
        val t1Expanded = tensor1.expand(expandBatchPortion + t1MatrixShape).view(listOf(-1) + t1MatrixShape)
        val t2Expanded = t2.expand(expandBatchPortion + t2MatrixShape).view(listOf(-1) + t2MatrixShape)

        val resShape = expandBatchPortion + listOf(t1MatrixShape[0], t2MatrixShape[1])

        // batched matmul on the single batch dimension tensors, than reshape to the broadcast shape
        var res = t1Expanded.bmm(t2Expanded).view(resShape)

        // if one of the input tensors were vectors, then the unsqueezed dimension must be squeezed in the result
        if (dim1 == 1)
            res = res.squeeze(-2)
        else if (dim2 == 1)
            res = res.squeeze(-1)

        return res
    }

    override fun gemm(addMatrix: Tensor, matrix: Tensor, alpha: Float, beta: Float): Tensor {
        return GeneralMatrixMultiplication(ops)(
            inplaceSafe(addMatrix),
            inplaceSafe(this),
            inplaceSafe(matrix),
            alpha,
            beta
        )
    }

    override fun mm(matrix: Tensor): Tensor {
        // TODO shape may call the forward function for now, but after all shape calculations are done it will be calculated in their invoke.
        //  We could call the forward immediately, but then we would have to rely on the garbage collector to clean up the
        //  intermediate outputs. (There is no way to know if a object is referenced)
        val addTensorShape = inplaceSafe(this).shape.mmShape(inplaceSafe(matrix).shape)
        // The add matrix doesn't require grads, because it won't be used as a result, just on the storage level
        val addZeros = ops.createFillConst(addTensorShape, false, 0f)
        return gemm(addZeros, inplaceSafe(matrix), 1f, 0f)
    }

    override fun batchedGemm(addTensor: Tensor, tensor: Tensor, alpha: Float, beta: Float): Tensor {
        return BatchedGeneralMatrixMultiplication(ops)(
            inplaceSafe(addTensor),
            inplaceSafe(this),
            inplaceSafe(tensor),
            alpha,
            beta
        )
    }

    override fun bmm(tensor: Tensor): Tensor { // TODO continue moving shape calculations and checks
        val addZeros = ops.createFillConst(
            listOf(inplaceSafe(this).shape[0], inplaceSafe(this).shape[1], inplaceSafe(tensor).shape[2]),
            false,
            0f
        )
        return batchedGemm(addZeros, inplaceSafe(tensor), 1f, 0f)
    }

    override fun gemv(addVector: Tensor, vector: Tensor, alpha: Float, beta: Float): Tensor {
        return GeneralMatrixVectorMultiplication(ops)(
            inplaceSafe(addVector),
            inplaceSafe(this),
            inplaceSafe(vector),
            alpha,
            beta
        )
    }

    override fun mv(vector: Tensor): Tensor {
        val addTensorShape = inplaceSafe(this).shape.mvShape(inplaceSafe(vector).shape)
        val addZeros = ops.createFillConst(addTensorShape, false, 0f)
        return gemv(addZeros, inplaceSafe(vector), 1f, 0f)
    }

    override fun dot(vector: Tensor): Tensor {
        return Dot(ops)(inplaceSafe(this), inplaceSafe(vector))
    }

    override fun variance(axis: Int, keepDimensions: Boolean, unbiased: Boolean): Tensor {
        val (_, variance) = meanVariance(axis, keepDimensions, unbiased)
        return variance
    }

    override fun meanVariance(axis: Int, keepDimensions: Boolean, unbiased: Boolean): Pair<Tensor, Tensor> {
        val tensor = inplaceSafe(this)
        val mean = tensor.mean(axis, true)
        val elements = tensor.shape[tensor.shape.normalizeAxis(axis)] - if (unbiased) 1 else 0
        val zeroCentered = tensor - mean
        val variance = (zeroCentered pow 2).sum(axis, keepDimensions) / elements.toFloat()
        return mean to variance
    }

    override fun std(axis: Int, keepDimensions: Boolean, unbiased: Boolean): Tensor {
        val (_, std) = meanStd(axis, keepDimensions, unbiased)
        return std
    }

    override fun meanStd(axis: Int, keepDimensions: Boolean, unbiased: Boolean): Pair<Tensor, Tensor> {
        val (mean, variance) = meanVariance(axis, keepDimensions, unbiased)
        return mean to sqrt(variance)
    }

    override fun view(newShape: List<Int>): Tensor {
        return Reshape(ops, viewMode = true)(inplaceSafe(this), newShape)
    }

    override fun lt(tensor: Tensor): Tensor {
        return LessThan(ops)(inplaceSafe(this), inplaceSafe(tensor))
    }

    override fun lte(tensor: Tensor): Tensor {
        return LessThanEquals(ops)(inplaceSafe(this), inplaceSafe(tensor))
    }

    override fun gt(tensor: Tensor): Tensor {
        return GreaterThan(ops)(inplaceSafe(this), inplaceSafe(tensor))
    }

    override fun gte(tensor: Tensor): Tensor {
        return GreaterThanEquals(ops)(inplaceSafe(this), inplaceSafe(tensor))
    }

    override fun eq(tensor: Tensor): Tensor {
        return Equals(ops)(inplaceSafe(this), inplaceSafe(tensor))
    }

    override fun neq(tensor: Tensor): Tensor {
        return NotEquals(ops)(inplaceSafe(this), inplaceSafe(tensor))
    }

    override fun lt(value: Float): Tensor {
        return LessThanConstant(ops)(inplaceSafe(this), value)
    }

    override fun lte(value: Float): Tensor {
        return LessThanEqualsConstant(ops)(inplaceSafe(this), value)
    }

    override fun gt(value: Float): Tensor {
        return GreaterThanConstant(ops)(inplaceSafe(this), value)
    }

    override fun gte(value: Float): Tensor {
        return GreaterThanEqualsConstant(ops)(inplaceSafe(this), value)
    }

    override fun eq(value: Float): Tensor {
        return EqualsConstant(ops)(inplaceSafe(this), value)
    }

    override fun neq(value: Float): Tensor {
        return NotEqualsConstant(ops)(inplaceSafe(this), value)
    }

    internal abstract fun backwardWithGrad(gradient: AbstractRawTensor<Any>)

    override fun noGrad(): Tensor {
        val tensor = inplaceSafe(this)
        tensor.forward()
        return NoGradVariable(ops, tensor.getRawValue())
    }

    override fun retainGrad(): Tensor {
        return RetainGrad(ops)(inplaceSafe(this))
    }

    override fun asVariable(requiresGrad: Boolean): Tensor {
        val safeThis = inplaceSafeForwarded()

        if (requiresGrad) {
            if (safeThis is RetainGrad && safeThis.savedGradient != null)
                return Variable(ops, safeThis.getRawValue(), safeThis.savedGradient!!)

            return Variable(ops, safeThis.getRawValue())
        }

        return NoGradVariable(ops, safeThis.getRawValue())
    }

    override fun platformOps(): TensorOperations<AbstractRawTensor<Any>> = ops

    override fun serialize(): CommonSerializableTensorDescriptor {
        return asVariable(true).serialize()
    }

    fun gradientAggregate(calculation: (Tensor) -> Tensor): Tensor {
        return GradientAggregatorFunction(ops)(inplaceSafe(this), calculation)
    }

    fun gradientAggregate(shape: List<Int>, calculation: (Tensor) -> Tensor): Tensor {
        return GradientAggregatorFunction(ops)(inplaceSafe(this), calculation, shape)
    }

}