package eu.redbean.kten.jvm.tensor.store

import eu.redbean.kten.api.autograd.utils.*
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.store.*
import eu.redbean.kten.jvm.tensor.operations.MemLeakDetectingJVMTensorOperations
import java.util.stream.IntStream
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

class JVMRawTensor(
    shape: List<Int>,
    store: FloatArray,
    val platformKey: String
): AbstractRawTensor<FloatArray>(shape, store) {

    private var refCount = 1
    internal var mustSurviveGC = false

    init {
        if (platformKey == MemLeakDetectingJVMTensorOperations.platformKey) {
            MemLeakDetectingJVMTensorOperations.rawMemUsage.addAndGet(storeReference.size * 4L)
            MemLeakDetectingJVMTensorOperations.instanceCollector.invoke(this)
        }
    }

    internal fun incrementRef() {
        if (platformKey == MemLeakDetectingJVMTensorOperations.platformKey)
            refCount++
    }

    internal fun release(): Boolean {
        if (platformKey == MemLeakDetectingJVMTensorOperations.platformKey) {
            if (!mustSurviveGC)
                refCount--

            if (refCount == 0) {
                storeReference = FloatArray(0)
                return true
            }
        }
        return false
    }

    override fun getSingleValue(indexes: List<Int>): Float {
        return this.storeReference[this.shape.tensorIndexing(indexes).storeIndex]
    }

    override fun broadcastTo(shape: List<Int>): JVMRawTensor {
        if (this.shape == shape)
            return this

        this.shape.inferExplicitBroadcastShape(shape)

        val thisShape = this.shape.toMutableList()
        for (i in 0 until (shape.size - thisShape.size)) {
            thisShape.add(0, 1)
        }

        val targetStoreIndexing = StoreIndexing(shape)

        val elements = FloatArray(shape.toStoreSize()) {
            val indices = targetStoreIndexing.toTensorIndex(it)
            for (i in 0 until indices.size) {
                indices[i] %= thisShape[i]
            }
            this.storeReference[thisShape.tensorIndexing(indices, false).storeIndex]
        }

        return JVMRawTensor(shape, elements, platformKey)
    }

    private fun elementwiseOpOnCommonShapedTensors(other: AbstractRawTensor<FloatArray>, op: (Float, Float) -> Float): JVMRawTensor {
        return JVMRawTensor(this.shape, FloatArray(this.storeReference.size) {
            op(this.storeReference[it], other.storeReference[it])
        }, platformKey)
    }

    private fun elementwiseOpOnTensors(other: AbstractRawTensor<FloatArray>, op: (Float, Float) -> Float): JVMRawTensor {
        val commonShape = calculateCommonShapeWith(other)
        val t1 = this.broadcastTo(commonShape)
        val t2 = other.broadcastTo(commonShape)

        return t1.elementwiseOpOnCommonShapedTensors(t2, op)
    }

    private fun inplaceElementwiseOpOnTensors(other: AbstractRawTensor<FloatArray>, op: (Float, Float) -> Float) {
        if (this.shape != other.shape)
            throw IllegalArgumentException(
                "Shapes does not match for inplace operation on tensors with shapes: ${this.shape} and ${other.shape}."
            )

        IntStream.range(0, this.storeReference.size).parallel()
            .forEach { this.storeReference[it] = op(this.storeReference[it], other.storeReference[it]) }
    }

    private fun mapElementsCopy(mapping: (Float) -> Float): JVMRawTensor {
        return JVMRawTensor(this.shape, FloatArray(this.storeReference.size) {
            mapping(this.storeReference[it])
        }, platformKey)
    }

    private fun inplaceMapElements(mapping: (Float) -> Float) {
        IntStream.range(0, this.storeReference.size).parallel()
            .forEach { this.storeReference[it] = mapping(this.storeReference[it]) }
    }

    override fun plus(other: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> = elementwiseOpOnTensors(other, Float::plus)

    override fun plus(constant: Float): AbstractRawTensor<FloatArray> = mapElementsCopy { it + constant }

    override fun plusAssign(other: AbstractRawTensor<FloatArray>) = inplaceElementwiseOpOnTensors(other, Float::plus)

    override fun plusAssign(constant: Float) = inplaceMapElements { it + constant }

    override fun minus(other: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> = elementwiseOpOnTensors(other, Float::minus)

    override fun minus(constant: Float): AbstractRawTensor<FloatArray> = mapElementsCopy { it - constant }

    override fun unaryMinus(): AbstractRawTensor<FloatArray> = mapElementsCopy { -it }

    override fun times(other: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> = elementwiseOpOnTensors(other, Float::times)

    override fun timesAssign(other: AbstractRawTensor<FloatArray>) = inplaceElementwiseOpOnTensors(other, Float::times)

    override fun times(constant: Float): AbstractRawTensor<FloatArray> =mapElementsCopy { it * constant }

    override fun timesAssign(constant: Float) = inplaceMapElements { it * constant }

    override fun div(other: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> = elementwiseOpOnTensors(other, Float::div)

    override fun div(constant: Float): AbstractRawTensor<FloatArray> = mapElementsCopy { it / constant }

    override fun divAssign(constant: Float) = inplaceMapElements { it / constant }

    override fun pow(other: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> = elementwiseOpOnTensors(other, Float::pow)

    override fun pow(constant: Float): AbstractRawTensor<FloatArray> = mapElementsCopy { it.pow(constant) }

    override fun reciprocal(): AbstractRawTensor<FloatArray> = mapElementsCopy { 1.0f / it }

    override fun log(): AbstractRawTensor<FloatArray> = mapElementsCopy { ln(it) }

    override fun sum(): AbstractRawTensor<FloatArray> {
        return JVMRawTensor(listOf(1), FloatArray(1) { storeReference.sum() }, platformKey)
    }

    override fun mean(): AbstractRawTensor<FloatArray> {
        return JVMRawTensor(listOf(1), FloatArray(1) { storeReference.average().toFloat() }, platformKey)
    }

    private fun mapAggregatingOp(
        aggregatingOperation: AggregatingOperation,
        axis: Int
    ): (IntArray) -> Float {
        val op: (Float, Float) -> Float = when (aggregatingOperation) {
            AggregatingOperation.MAX -> { e1, e2 -> kotlin.math.max(e1, e2) }
            AggregatingOperation.MIN -> { e1, e2 -> kotlin.math.min(e1, e2) }
            AggregatingOperation.SUM -> { e1, e2 -> e1 + e2 }
            else -> { e1, e2 -> Float.NaN }
        }

        return when (aggregatingOperation) {
            AggregatingOperation.ARG_MAX,
            AggregatingOperation.ARG_MIN -> { indexes ->
                var tempVal = when (aggregatingOperation) {
                    AggregatingOperation.ARG_MAX -> Float.NEGATIVE_INFINITY
                    else -> Float.POSITIVE_INFINITY
                }
                var aggVal = 0.0f
                for (idx in 0 until this.shape[axis]) {
                    indexes[axis] = idx
                    val currentVal = getSingleValue(indexes.toList())
                    if ((aggregatingOperation == AggregatingOperation.ARG_MAX && currentVal > tempVal)
                        || (aggregatingOperation == AggregatingOperation.ARG_MIN && currentVal < tempVal)
                    ) {
                        tempVal = currentVal
                        aggVal = idx.toFloat()
                    }
                }
                aggVal
            }
            AggregatingOperation.MEAN -> { indexes ->
                var aggVal = 0.0f
                for (j in 0 until this.shape[axis]) {
                    indexes[axis] = j
                    aggVal += getSingleValue(indexes.toList())
                }
                aggVal / this.shape[axis]
            }
            else -> { indexes ->
                var aggVal = when (aggregatingOperation) {
                    AggregatingOperation.MAX -> Float.NEGATIVE_INFINITY
                    AggregatingOperation.MIN -> Float.POSITIVE_INFINITY
                    else -> 0.0f
                }
                for (j in 0 until this.shape[axis]) {
                    indexes[axis] = j
                    aggVal = op(
                        aggVal,
                        getSingleValue(indexes.toList())
                    )
                }
                aggVal
            }
        }
    }

    override fun aggregateOverAxis(axis: Int, resShape: List<Int>, aggregatingOperation: AggregatingOperation): AbstractRawTensor<FloatArray> {
        val resSize = resShape.toStoreSize()
        val elements = FloatArray(resSize)

        val opForIndexes = mapAggregatingOp(aggregatingOperation, axis)

        val targetStoreIndexing = StoreIndexing(resShape)

        IntStream.range(0, resSize).parallel()
            .forEach {
                elements[it] = opForIndexes(targetStoreIndexing.toTensorIndex(it))
            }

        return JVMRawTensor(resShape, elements, platformKey)
    }

    private fun internalInplaceScatter(axis: Int, index: AbstractRawTensor<FloatArray>, src: AbstractRawTensor<FloatArray>,
                                       addSameIndex: Boolean = false) {
        val normalizedAxis = shape.normalizedAxisScatter(axis, index.shape, src.shape)

        val indexSizeAtAxis = index.shape[normalizedAxis]
        val thisSizeAtAxis = shape[normalizedAxis]

        index.storeIndexing.applyAt(normalizedAxis) { indexMapping ->
            for (i in 0 until indexSizeAtAxis) {
                val idx = index.storeReference[indexMapping(i, index.storeIndexing)].toInt()

                if (idx < 0 || idx >= thisSizeAtAxis)
                    throw IllegalArgumentException("Scatter got invalid index, index value $idx is out of bounds 0 and ${thisSizeAtAxis} at axis: $axis")

                if (addSameIndex)
                    this.storeReference[indexMapping(idx, this.storeIndexing)] += src.storeReference[indexMapping(i, src.storeIndexing)]
                else
                    this.storeReference[indexMapping(idx, this.storeIndexing)] = src.storeReference[indexMapping(i, src.storeIndexing)]
            }
        }
    }

    override fun inplaceScatter(axis: Int, index: AbstractRawTensor<FloatArray>, src: AbstractRawTensor<FloatArray>) {
        internalInplaceScatter(axis, index, src)
    }

    override fun inplaceScatterAdd(axis: Int, index: AbstractRawTensor<FloatArray>, src: AbstractRawTensor<FloatArray>) {
        internalInplaceScatter(axis, index, src, addSameIndex = true)
    }

    override fun copy(shallow: Boolean): AbstractRawTensor<FloatArray> {
        if (shallow) {
            return JVMRawTensor(shape, storeReference, platformKey)
        }
        return JVMRawTensor(this.shape, this.storeReference.copyOf(), platformKey)
    }

    override fun view(shape: List<Int>): AbstractRawTensor<FloatArray> {
        val res = JVMRawTensor(this.shape.reshape(shape), this.storeReference, platformKey)
        res.incrementRef()
        return res
    }

    override fun get(index: IntArray): AbstractRawTensor<FloatArray> {
        val (normIndex, resShape) = this.shape.normalizedIndexedShape(index)
        val range = SimpleTensorIndexing(this.shape, normIndex).storeRange
        return JVMRawTensor(resShape, this.storeReference.sliceArray(range), platformKey)
    }

    override fun get(index: Array<out IntRange>): AbstractRawTensor<FloatArray> {
        val (normIndex, resShape) = this.shape.normalizedIndexedShape(index)
        val sparseRanges = RangeTensorIndexing(this.shape, normIndex).sparseStoreRanges
        return JVMRawTensor(resShape, sparseRanges.map { this.storeReference.sliceArray(it) }.reduce(FloatArray::plus), platformKey)
    }

    override fun set(index: IntArray, value: AbstractRawTensor<FloatArray>) {
        val range = this.shape.tensorIndexing(index).storeRange
        if (value.storeReference.size != (range.last - range.first + 1))
            throw IllegalArgumentException("Cannot set tensor with shape: ${value.shape} to tensor with shape: ${this.shape} at index: ${index.toList()}")
        range.forEachIndexed { valueIndex, storeIndex ->
            this.storeReference[storeIndex] = value.storeReference[valueIndex]
        }
    }

    override fun set(index: Array<out IntRange>, value: AbstractRawTensor<FloatArray>) {
        val sparseRanges = this.shape.tensorIndexing(index).sparseStoreRanges
        val size = sparseRanges.map { it.last - it.first + 1 }.sum()

        if (size != value.storeReference.size)
            throw IllegalArgumentException("Cannot set tensor with shape ${value.shape} to tensor with shape: ${this.shape} at index: ${index.toList()}")

        var valueIndex = 0
        sparseRanges.forEach { range ->
            range.forEach {
                this.storeReference[it] = value.storeReference[valueIndex++]
            }
        }
    }

    override fun set(index: IntArray, value: Float) {
        val range = this.shape.tensorIndexing(index).storeRange
        this.storeReference.fill(value, range.first, range.last + 1)
    }

    override fun set(index: Array<out IntRange>, value: Float) {
        this.shape.tensorIndexing(index).sparseStoreRanges.forEach {
            this.storeReference.fill(value, it.first, it.last + 1)
        }
    }

    override fun inplaceScatter(axis: Int, index: AbstractRawTensor<FloatArray>, src: Float) {
        val normAxis = shape.normalizeAxis(axis) // TODO check shape compatibility for target and index
        val indexSizeAtAxis = index.shape[normAxis]
        val thisSizeAtAxis = this.shape[normAxis]

        index.storeIndexing.applyAt(normAxis) { indexMapping ->
            for (i in 0 until indexSizeAtAxis) {
                val idx = index.storeReference[indexMapping(i, index.storeIndexing)].toInt()

                if (idx < 0 || idx >= thisSizeAtAxis)
                    throw IllegalArgumentException("Scatter got invalid index, index value $idx is out of bounds 0 and ${thisSizeAtAxis} at axis: $axis")

                this.storeReference[indexMapping(idx, this.storeIndexing)] = src
            }
        }
    }

    override fun gather(axis: Int, index: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> {
        val normAxis = shape.normalizeAxis(axis) // TODO check shape compatibility
        val res = JVMRawTensor(index.shape.toList(), FloatArray(index.shape.toStoreSize()), platformKey)

        val indexSizeAtAxis = index.shape[normAxis]
        val thisSizeAtAxis = this.shape[normAxis]

        res.storeIndexing.applyAt(normAxis) { indexMapping ->
            for (i in 0 until indexSizeAtAxis) {
                val idx = index.storeReference[indexMapping(i, index.storeIndexing)].toInt()

                if  (idx < 0 || idx >= thisSizeAtAxis)
                    throw IllegalArgumentException("Gather got invalid index, index value $idx is out of bounds 0 and ${thisSizeAtAxis} at axis: $axis")

                res.storeReference[indexMapping(i, res.storeIndexing)] = this.storeReference[indexMapping(idx, this.storeIndexing)]
            }
        }

        return res
    }

    override fun lt(tensor: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> = elementwiseOpOnCommonShapedTensors(tensor) { a, b ->
        if (a < b) 1.0f else 0.0f
    }

    override fun lte(tensor: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> = elementwiseOpOnCommonShapedTensors(tensor) { a, b ->
        if (a <= b) 1.0f else 0.0f
    }

    override fun gt(tensor: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> = elementwiseOpOnCommonShapedTensors(tensor) { a, b ->
        if (a > b) 1.0f else 0.0f
    }

    override fun gte(tensor: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> = elementwiseOpOnCommonShapedTensors(tensor) { a, b ->
        if (a >= b) 1.0f else 0.0f
    }

    override fun eq(tensor: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> = elementwiseOpOnCommonShapedTensors(tensor) { a, b ->
        if (kotlin.math.abs(a - b) < PlatformProvider.epsilon) 1.0f else 0.0f
    }

    override fun neq(tensor: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> = elementwiseOpOnCommonShapedTensors(tensor) { a, b ->
        if (kotlin.math.abs(a - b) < PlatformProvider.epsilon) 0.0f else 1.0f
    }

    override fun lt(value: Float): AbstractRawTensor<FloatArray> = mapElementsCopy { if (it < value) 1.0f else 0.0f }

    override fun lte(value: Float): AbstractRawTensor<FloatArray> = mapElementsCopy { if (it <= value) 1.0f else 0.0f }

    override fun gt(value: Float): AbstractRawTensor<FloatArray> = mapElementsCopy { if (it > value) 1.0f else 0.0f }

    override fun gte(value: Float): AbstractRawTensor<FloatArray> = mapElementsCopy { if (it >= value) 1.0f else 0.0f }

    override fun eq(value: Float): AbstractRawTensor<FloatArray> = mapElementsCopy {
        if (kotlin.math.abs(it - value) < PlatformProvider.epsilon) 1.0f else 0.0f
    }

    override fun neq(value: Float): AbstractRawTensor<FloatArray> = mapElementsCopy {
        if (kotlin.math.abs(it - value) < PlatformProvider.epsilon) 0.0f else 1.0f
    }

    override fun exp(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.exp(it) }

    override fun tanh(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.tanh(it) }

    override fun sigmoid(): AbstractRawTensor<FloatArray> = mapElementsCopy { 1.0f / (1.0f + kotlin.math.exp(-1.0f * it)) }

    override fun sinh(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.sinh(it) }

    override fun cosh(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.cosh(it) }

    override fun abs(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.abs(it) }

    override fun sign(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.sign(it) }

    override fun clamp(min: Float, max: Float): AbstractRawTensor<FloatArray> = mapElementsCopy {
        min(max(it, min), max)
    }

    override fun sqrt(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.sqrt(it) }

    override fun sin(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.sin(it) }

    override fun cos(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.cos(it) }

    override fun tan(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.tan(it) }

    override fun asin(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.asin(it) }

    override fun acos(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.acos(it) }

    override fun atan(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.atan(it) }

    override fun floor(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.floor(it) }

    override fun ceil(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.ceil(it) }

    override fun round(): AbstractRawTensor<FloatArray> = mapElementsCopy { Math.round(it).toFloat() }

    override fun trunc(): AbstractRawTensor<FloatArray> = mapElementsCopy { kotlin.math.truncate(it) }

    override fun rsqrt(): AbstractRawTensor<FloatArray> = mapElementsCopy { 1.0f / kotlin.math.sqrt(it) }

    override fun transpose(axis1: Int, axis2: Int): AbstractRawTensor<FloatArray> {
        val (newShape, normAxis1, normAxis2) = shape.transposeNormalizeAxes(axis1, axis2)
        val res = this.copy()
        res.shape = newShape
        val rangesThis = shape.map { 0 until it }.toTypedArray()
        val rangesRes = newShape.map { 0 until it }.toTypedArray()

        for (i in 0 until shape[normAxis1]) {
            rangesThis[normAxis1] = i..i
            rangesRes[normAxis2] = i..i
            res[rangesRes] = this[rangesThis]
        }

        return res
    }

    override fun permute(axes: List<Int>): AbstractRawTensor<FloatArray> {
        val newShape = shape.permute(axes)
        val res = this.copy()
        res.shape = newShape

        IntStream.range(0, storeReference.size).parallel().forEach {
            val tensorIndex = storeIndexing.toTensorIndex(it)
            val byAxesIndex = IntArray(tensorIndex.size) { tensorIndex[axes[it]] }

            res.storeReference[newShape.tensorIndexing(byAxesIndex).storeIndex] = this.storeReference[it]
        }

        return res
    }

    override fun narrow(axis: Int, start: Int, length: Int): AbstractRawTensor<FloatArray> {
        val normAxis = shape.normalizeAxis(axis) // TODO check start + length < shape[axis] in tensor impl.
        val ranges = shape.map { 0 until it }.toTypedArray()
        ranges[normAxis] = start until (start + length)
        return this[ranges]
    }

    override fun dot(vector: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> {
        val resShape = shape.dotShape(vector.shape)
        val resValue = this.storeReference.zip(vector.storeReference)
            .map { it.first * it.second }
            .reduce { a, b -> a + b }

        return JVMRawTensor(resShape, FloatArray(1) { resValue }, platformKey)
    }

    override fun indexSelect(axis: Int, index: AbstractRawTensor<FloatArray>): AbstractRawTensor<FloatArray> {
        val (normAxis, resShape) = shape.indexSelectNormAxisShape(axis, index.shape)
        val indexSize = index.shape[0]
        val thisSize = this.shape[normAxis]

        val res = JVMRawTensor(resShape, FloatArray(resShape.toStoreSize()), platformKey)

        res.storeIndexing.applyAt(normAxis) { indexMapping ->
            for (i in 0 until indexSize) {
                val idx = index.storeReference[i].toInt()
                if (idx < 0 || idx >= thisSize) {
                    throw IllegalArgumentException("Index select got invalid index, index value $idx is out of bounds 0 and ${thisSize} at axis: $axis")
                }
                res.storeReference[indexMapping(i, res.storeIndexing)] = this.storeReference[indexMapping(idx, this.storeIndexing)]
            }
        }

        return res
    }

    override fun indexAdd(axis: Int, index: AbstractRawTensor<FloatArray>, src: AbstractRawTensor<FloatArray>, alpha: Float) {
        val normAxis = shape.indexAddCheckShapesNormAxis(axis, index.shape, src.shape)
        val indexSize = index.shape[0]
        val thisSize = shape[normAxis]

        this.storeIndexing.applyAt(normAxis) { indexMapping ->
            for (i in 0 until indexSize) {
                val idx = index.storeReference[i].toInt()
                if (idx < 0 || idx >= thisSize) {
                    throw IllegalArgumentException("Index Add got invalid index, index value $idx is out of bounds 0 and ${thisSize} at axis: $axis")
                }
                this.storeReference[indexMapping(idx, this.storeIndexing)] += src.storeReference[indexMapping(i, src.storeIndexing)] * alpha
            }
        }
    }

    override fun containsNan(): Boolean {
        return storeReference.any { it.isNaN() }
    }

    override fun inplaceResize(vararg shape: Int) {
        inplaceResize(shape.toList())
    }

    fun inplaceResize(newShape: List<Int>) {
        storeReference = storeReference.copyOf(newShape.toStoreSize())
        this.shape = newShape
    }

    override fun inplaceFill(value: Float) {
        storeReference.fill(value)
    }

    fun getView(vararg index: Int): JVMRawTensorView {
        val (normIndex, resShape) = this.shape.normalizedIndexedShape(index)
        return JVMRawTensorView(resShape, StoreView(SimpleTensorIndexing(shape, normIndex).storeRange, storeReference))
    }

    fun view(vararg newShape: Int): JVMRawTensorView {
        return JVMRawTensorView(shape.reshape(newShape.toList()), StoreView(0 until storeReference.size, storeReference))
    }

    fun asView(): JVMRawTensorView {
        return JVMRawTensorView(shape, StoreView(0 until storeReference.size, storeReference))
    }

}