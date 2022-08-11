package eu.redbean.kten.api.tensor.store

import eu.redbean.kten.api.autograd.utils.*

abstract class AbstractRawTensor<STORE_TYPE>(
    shape: List<Int>,
    storeReference: STORE_TYPE
) {

    var storeReference: STORE_TYPE = storeReference
        protected set

    var shape = shape
        get() = field
        set(value) {
            field = value
            storeIndexing = StoreIndexing(value)
        }

    val dimensions: Int
        get() = shape.size

    var storeIndexing = StoreIndexing(shape)

    abstract fun broadcastTo(shape: List<Int>): AbstractRawTensor<STORE_TYPE>

    abstract operator fun get(index: IntArray): AbstractRawTensor<STORE_TYPE>

    abstract operator fun get(index: Array<out IntRange>): AbstractRawTensor<STORE_TYPE>

    abstract operator fun set(index: IntArray, value: AbstractRawTensor<STORE_TYPE>)

    abstract operator fun set(index: Array<out IntRange>, value: AbstractRawTensor<STORE_TYPE>)

    abstract operator fun set(index: IntArray, value: Float)

    abstract operator fun set(index: Array<out IntRange>, value: Float)

    abstract operator fun plus(other: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>

    abstract operator fun plus(constant: Float): AbstractRawTensor<STORE_TYPE>

    abstract operator fun plusAssign(other: AbstractRawTensor<STORE_TYPE>)

    abstract operator fun plusAssign(constant: Float)

    abstract operator fun minus(other: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>

    abstract operator fun minus(constant: Float): AbstractRawTensor<STORE_TYPE>

    abstract operator fun unaryMinus(): AbstractRawTensor<STORE_TYPE>

    abstract operator fun times(other: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>

    abstract operator fun timesAssign(other: AbstractRawTensor<STORE_TYPE>)

    abstract operator fun times(constant: Float): AbstractRawTensor<STORE_TYPE>

    abstract operator fun timesAssign(constant: Float)

    abstract operator fun div(other: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>

    abstract operator fun div(constant: Float): AbstractRawTensor<STORE_TYPE>

    abstract operator fun divAssign(constant: Float)

    abstract infix fun pow(other: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>

    abstract infix fun pow(constant: Float): AbstractRawTensor<STORE_TYPE>

    abstract fun reciprocal(): AbstractRawTensor<STORE_TYPE>

    abstract fun log(): AbstractRawTensor<STORE_TYPE>

    protected abstract fun aggregateOverAxis(axis: Int, resShape: List<Int>, aggregatingOperation: AggregatingOperation): AbstractRawTensor<STORE_TYPE>

    fun reshape(vararg shape: Int): AbstractRawTensor<STORE_TYPE> = reshape(shape.toList())

    fun reshape(shape: List<Int>): AbstractRawTensor<STORE_TYPE> {
        val newShape = this.shape.reshape(shape)
        val res = this.copy()
        res.shape = newShape
        return res
    }

    fun inplaceReshape(shape: List<Int>) {
        this.shape = this.shape.reshape(shape)
    }

    abstract fun view(shape: List<Int>): AbstractRawTensor<STORE_TYPE>

    private fun aggregatingOpOverAxis(
        axis: Int,
        keepDims: Boolean,
        aggregatingOperation: AggregatingOperation
    ): AbstractRawTensor<STORE_TYPE> {
        val normalizedAxis = shape.normalizeAxis(axis)
        val resShape = shape.aggregateOver(normalizedAxis, true)
        val res = aggregateOverAxis(normalizedAxis, resShape, aggregatingOperation)

        if (keepDims.not() && resShape.size > 1) {
            val shapeWithoutExtraDim = resShape.toMutableList()
            shapeWithoutExtraDim.removeAt(normalizedAxis)
            res.shape = shapeWithoutExtraDim
        }

        return res
    }


    fun sum(axis: Int, keepDimensions: Boolean = false): AbstractRawTensor<STORE_TYPE> = aggregatingOpOverAxis(axis, keepDimensions, AggregatingOperation.SUM)

    fun mean(axis: Int, keepDimensions: Boolean = false): AbstractRawTensor<STORE_TYPE> = aggregatingOpOverAxis(axis, keepDimensions, AggregatingOperation.MEAN)

    fun max(axis: Int, keepDimensions: Boolean = false): AbstractRawTensor<STORE_TYPE> = aggregatingOpOverAxis(axis, keepDimensions, AggregatingOperation.MAX)
    fun min(axis: Int, keepDimensions: Boolean = false): AbstractRawTensor<STORE_TYPE> = aggregatingOpOverAxis(axis, keepDimensions, AggregatingOperation.MIN)

    fun argMax(axis: Int = 0, keepDimensions: Boolean = false): AbstractRawTensor<STORE_TYPE> = aggregatingOpOverAxis(axis, keepDimensions, AggregatingOperation.ARG_MAX)
    fun argMin(axis: Int = 0, keepDimensions: Boolean = false): AbstractRawTensor<STORE_TYPE> = aggregatingOpOverAxis(axis, keepDimensions, AggregatingOperation.ARG_MIN)

    abstract fun sum(): AbstractRawTensor<STORE_TYPE>
    abstract fun mean(): AbstractRawTensor<STORE_TYPE>

    internal fun inplaceSqueeze(axis: Int) {
        val newShape = this.shape.toMutableList()
        newShape.removeAt(shape.normalizeAxis(axis))
        this.shape = newShape
    }

    fun squeeze(axis: Int): AbstractRawTensor<STORE_TYPE> {
        val res = this.copy()
        res.inplaceSqueeze(axis)
        return res
    }

    internal fun inplaceUnsqueeze(axis: Int) {
        this.shape = this.shape.unsqueeze(axis)
    }

    fun unsqueeze(axis: Int): AbstractRawTensor<STORE_TYPE> {
        val res = this.copy()
        res.inplaceUnsqueeze(axis)
        return res
    }

    abstract fun inplaceScatter(axis: Int, index: AbstractRawTensor<STORE_TYPE>, src: AbstractRawTensor<STORE_TYPE>)

    abstract fun inplaceScatterAdd(axis: Int, index: AbstractRawTensor<STORE_TYPE>, src: AbstractRawTensor<STORE_TYPE>)

    abstract fun inplaceScatter(axis: Int, index: AbstractRawTensor<STORE_TYPE>, src: Float)

    fun scatter(axis: Int, index: AbstractRawTensor<STORE_TYPE>, src: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE> {
        val res = this.copy()
        res.inplaceScatter(axis, index, src)
        return res
    }

    fun scatter(axis: Int, index: AbstractRawTensor<STORE_TYPE>, src: Float): AbstractRawTensor<STORE_TYPE> {
        val res = this.copy()
        res.inplaceScatter(axis, index, src)
        return res
    }

    abstract fun gather(axis: Int, index: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>

    abstract fun getSingleValue(indexes: List<Int>): Float

    abstract fun copy(shallow: Boolean = false): AbstractRawTensor<STORE_TYPE>

    protected fun calculateCommonShapeWith(other: AbstractRawTensor<STORE_TYPE>): List<Int> {
        if (this.shape == other.shape) {
            return this.shape
        }

        var thisShape = this.shape
        var otherShape = other.shape

        val resShape = mutableListOf<Int>()

        val dimDiff = this.dimensions - other.dimensions
        if (dimDiff > 0) {
            resShape.addAll((0 until dimDiff).map { thisShape[it] })
            thisShape = thisShape.drop(dimDiff)
        } else if (dimDiff < 0) {
            resShape.addAll((0 until -dimDiff).map { otherShape[it] })
            otherShape = otherShape.drop(-dimDiff)
        }

        resShape.addAll(thisShape.zip(otherShape).map { kotlin.math.max(it.first, it.second) })

        return resShape
    }

    abstract infix fun lt(tensor: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>
    abstract infix fun lte(tensor: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>
    abstract infix fun gt(tensor: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>
    abstract infix fun gte(tensor: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>
    abstract infix fun eq(tensor: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>
    abstract infix fun neq(tensor: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>

    abstract infix fun lt(value: Float): AbstractRawTensor<STORE_TYPE>
    abstract infix fun lte(value: Float): AbstractRawTensor<STORE_TYPE>
    abstract infix fun gt(value: Float): AbstractRawTensor<STORE_TYPE>
    abstract infix fun gte(value: Float): AbstractRawTensor<STORE_TYPE>
    abstract infix fun eq(value: Float): AbstractRawTensor<STORE_TYPE>
    abstract infix fun neq(value: Float): AbstractRawTensor<STORE_TYPE>

    abstract fun exp(): AbstractRawTensor<STORE_TYPE>
    abstract fun tanh(): AbstractRawTensor<STORE_TYPE>
    abstract fun sigmoid(): AbstractRawTensor<STORE_TYPE>
    abstract fun sinh(): AbstractRawTensor<STORE_TYPE>
    abstract fun cosh(): AbstractRawTensor<STORE_TYPE>
    abstract fun abs(): AbstractRawTensor<STORE_TYPE>
    abstract fun sign(): AbstractRawTensor<STORE_TYPE>
    abstract fun clamp(min: Float, max: Float): AbstractRawTensor<STORE_TYPE>
    abstract fun sqrt(): AbstractRawTensor<STORE_TYPE>
    abstract fun sin(): AbstractRawTensor<STORE_TYPE>
    abstract fun cos(): AbstractRawTensor<STORE_TYPE>
    abstract fun tan(): AbstractRawTensor<STORE_TYPE>
    abstract fun asin(): AbstractRawTensor<STORE_TYPE>
    abstract fun acos(): AbstractRawTensor<STORE_TYPE>
    abstract fun atan(): AbstractRawTensor<STORE_TYPE>
    abstract fun floor(): AbstractRawTensor<STORE_TYPE>
    abstract fun ceil(): AbstractRawTensor<STORE_TYPE>
    abstract fun round(): AbstractRawTensor<STORE_TYPE>
    abstract fun trunc(): AbstractRawTensor<STORE_TYPE>
    abstract fun rsqrt(): AbstractRawTensor<STORE_TYPE>
    abstract fun transpose(axis1: Int, axis2: Int): AbstractRawTensor<STORE_TYPE>

    abstract fun permute(axes: List<Int>): AbstractRawTensor<STORE_TYPE>

    abstract fun narrow(axis: Int, start: Int, length: Int): AbstractRawTensor<STORE_TYPE>

    abstract fun dot(vector: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>

    abstract fun indexSelect(axis: Int, index: AbstractRawTensor<STORE_TYPE>): AbstractRawTensor<STORE_TYPE>

    abstract fun indexAdd(axis: Int, index: AbstractRawTensor<STORE_TYPE>, src: AbstractRawTensor<STORE_TYPE>, alpha: Float = 1f)

    abstract fun containsNan(): Boolean

    abstract fun inplaceResize(vararg shape: Int)

    abstract fun inplaceFill(value: Float)

    abstract fun maskedFill(mask: AbstractRawTensor<STORE_TYPE>, value: Float): AbstractRawTensor<STORE_TYPE>

}