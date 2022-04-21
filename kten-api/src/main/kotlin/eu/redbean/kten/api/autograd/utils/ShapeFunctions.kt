package eu.redbean.kten.api.autograd.utils

import eu.redbean.kten.api.tensor.Constants.all
import eu.redbean.kten.api.tensor.Constants.end
import eu.redbean.kten.api.tensor.Constants.start
import eu.redbean.kten.api.tensor.store.RangeTensorIndexing
import eu.redbean.kten.api.tensor.store.SimpleTensorIndexing
import java.util.*
import kotlin.math.max

fun List<Int>.reshape(newShape: List<Int>): List<Int> {
    if ((newShape.any { it < -1 }))
        throw IllegalArgumentException("Invalid shape: $newShape")

    val inferredDimensions = newShape.mapIndexed { index, i -> Pair(index, i) }.filter { (_, i) -> i == -1 }.map { it.first }

    if ((inferredDimensions.size > 1))
        throw IllegalArgumentException("Cannot reshape to ${newShape}. Only one dimension can be inferred")

    val resShape = newShape.toMutableList()

    val inferredDimensionIndex = inferredDimensions.firstOrNull()

    if (inferredDimensionIndex != null)
        resShape[inferredDimensionIndex] = 1

    val elements = this.fold(1, Int::times)
    var resShapeElements = resShape.fold(1, Int::times)

    if (inferredDimensionIndex != null) {
        resShape[inferredDimensionIndex] = elements / resShapeElements
        resShapeElements = resShape.fold(1, Int::times)
    }

    if (elements != resShapeElements)
        throw IllegalArgumentException("Cannot reshape tensor with shape: ${this} to shape: ${newShape}")

    return resShape.toList()
}

fun List<Int>.normalizeAxis(axis: Int): Int {
    val res = if (axis < 0) this.size + axis else axis

    if (res < 0 || res >= this.size)
        throw IllegalArgumentException("Cannot address axis: ${axis} in tensor with dimensions: ${this.size}")

    return res
}

fun List<Int>.squeeze(axis: Int): List<Int> {
    val normAxis = this.normalizeAxis(axis)

    if (this[normAxis] != 1)
        throw IllegalArgumentException("Cannot squeeze tensor with shape: ${this} at axis: ${axis}")

    val res = this.toMutableList()
    res.removeAt(normAxis)
    return res
}

fun List<Int>.unsqueeze(axis: Int): List<Int> {
    val normAxis = if (axis != this.size)
        if (axis < 0)
            (if (axis == (this.size + 1) * -1) 0 else this.normalizeAxis(axis) + 1)
        else this.normalizeAxis(axis)
    else axis

    val res = this.toMutableList()
    res.add(normAxis, 1)
    return res
}

fun List<Int>.aggregateOver(axis: Int, keepDimensions: Boolean): List<Int> {
    val normAxis = this.normalizeAxis(axis)
    val res = this.toMutableList()

    res[normAxis] = 1

    if (keepDimensions.not() && res.size > 1)
        res.removeAt(normAxis)

    return res
}

fun concatShapes(shapes: List<List<Int>>, axis: Int): List<Int> {
    if (shapes.map { it.size }.distinct().size == 1) {
        val normalizedAxis = shapes[0].normalizeAxis(axis)
        if (shapes.map {
                val shapeWithoutAxis = it.toMutableList()
                shapeWithoutAxis.removeAt(normalizedAxis)
                shapeWithoutAxis
            }.distinct().size == 1) {
            val newSizeAtAxis = shapes.map { it[normalizedAxis] }.sum()
            val res = shapes[0].toMutableList()
            res[normalizedAxis] = newSizeAtAxis
            return res
        }
    }

    throw IllegalArgumentException(
        "Cannot concatenate tensors with shapes: ${shapes} at axis: ${axis} " +
                "(All tensors must have the same number of dimenisons, and have the same size at all dimenions except at the dimension indexed by the axis)"
    )
}

fun List<Int>.permute(axes: List<Int>): List<Int> {
    if (this.size != axes.size || !axes.containsAll(this.indices.toList()))
        throw IllegalArgumentException("Tensor with shape: ${this} cannot be permuted by axes: ${axes}. All dimensions must be indexed.")

    return List(this.size) {
        this[axes[it]]
    }
}

fun List<Int>.transposeNormalizeAxes(axis1: Int, axis2: Int): Triple<List<Int>, Int, Int> {
    val normAxis1 = this.normalizeAxis(axis1)
    val normAxis2 = this.normalizeAxis(axis2)

    if (normAxis1 == normAxis2)
        throw IllegalArgumentException("Transpose requires different axes as parameters, " +
                "but got $axis1 (normalized: $normAxis1) and $axis2 (normalized: $normAxis2)")

    val res = this.toMutableList()
    res[normAxis1] = this[normAxis2]
    res[normAxis2] = this[normAxis1]

    return Triple(res, normAxis1, normAxis2)
}

operator fun List<Int>.get(range: IntRange): List<Int> {

    fun startOrEnd(value: Int): Int {
        if (value == start)
            return 0
        else if (value == end)
            return this.size - 1
        else
            return value
    }

    var first = range.first
    var last = range.endInclusive
    if (range == all) {
        first = 0
        last = this.size - 1
    } else {
        first = startOrEnd(first)
        first = if (first < 0) this.size + first else first

        last = startOrEnd(last)
        last = if (last < 0) this.size + last - 1 else last
    }

    if (last < first) {
        return this.reversed().subList(this.size - last, this.size - first + 1) //TODO check
    }

    return this.subList(first, last + 1)
}

operator fun List<Int>.times(repeat: Int): List<Int> {
    val res = mutableListOf<Int>()
    for (i in 0 until repeat)
        res.addAll(this)
    return res
}

fun inferImplicitBroadcastShape(shape1: List<Int>, shape2: List<Int>): List<Int> {
    val dimensions = if (shape1.size > shape2.size) shape1.size else shape2.size
    val expandShape1 = listOf(1) * (dimensions - shape1.size) + shape1
    val expandShape2 = listOf(1) * (dimensions - shape2.size) + shape2
    return inferBroadcastShapeOfAlignedShapes(expandShape1, expandShape2)
}

private fun inferBroadcastShapeOfAlignedShapes(shape1: List<Int>, shape2: List<Int>): List<Int> {
    return shape1.zip(shape2).map { (a, b) ->
        if (a == b || a == 1 || b == 1)
            max(a, b)
        else
            throw IllegalArgumentException(
                "Shape $shape1 must match $shape2 for every non-singleton dimensions (missing dimensions count as singleton " +
                        "dimension at the beginning of the shapes)"
            )
    }
}

fun List<Int>.inferExplicitBroadcastShape(shape: List<Int>): List<Int> {
    if (this.size > shape.size)
        throw IllegalArgumentException("Tensor with shape: ${this} cannot be broadcasted to shape: ${shape}")

    val expandedShape = listOf(1) * (shape.size - this.size) + this
    return expandedShape.zip(shape).map { (a, b) ->
        if (a == b || a == 1)
            max(a, b)
        else
            throw IllegalArgumentException(
                "Shape $this must match $shape for every non-singleton dimensions (missing dimensions count as singleton " +
                        "dimension at the beginning of the shapes)"
            )
    }
}

fun List<Int>.normalizeIndex(index: List<Int>): List<Int> {
    if (index.size > this.size)
        throw IllegalArgumentException("Cannot index tensor with shape: ${this} with index: ${index}")

    return index.mapIndexed { idx, item ->
        val size = this[idx]
        val normIndexItem = if (item < 0) size + item else item
        if (normIndexItem < 0 || normIndexItem >= size)
            throw IllegalArgumentException("Cannot index tensor with shape: ${this} with index: ${index}")

        normIndexItem
    }
}

fun List<Int>.normalizeIndex(index: IntArray): IntArray {
    if (index.size > this.size)
        throw IllegalArgumentException("Cannot index tensor with shape: ${this} with index: ${index}")

    return IntArray(index.size) { idx ->
        val size = this[idx]
        val item = index[idx]
        val normIndexItem = if (item < 0) size + item else item
        if (normIndexItem < 0 || normIndexItem >= size)
            throw IllegalArgumentException("Cannot index tensor with shape: ${this} with index: ${index}")

        normIndexItem
    }
}

fun List<Int>.normalizedIndexedShape(index: IntArray): Pair<IntArray, List<Int>> {
    val normalizedIndex = this.normalizeIndex(index)
    return Pair(
        normalizedIndex,
        if (normalizedIndex.size == this.size) listOf(1) else this.drop(normalizedIndex.size)
    )
}

fun List<Int>.normalizeIndex(index: Array<out IntRange>): Array<IntRange> {
    val normalizedShapeIndexMapping = this.zip(index.asIterable())
        .map { (max, range) ->
            if (range == all)
                Pair(max, 0 until max)
            else if (range.first < 0 || range.last < 0)
                Pair(max, (if (range.first < 0) max + range.first else range.first)..(if (range.last < 0) max + range.last else range.last))
            else
                Pair(max, range)
        }

    val normalizedRangeInvalid = normalizedShapeIndexMapping.any { (max, range) ->
        range.first < 0 || range.last < 0 || range.first >= max || range.last >= max || range.first > range.last
    }

    if (index.size > this.size || normalizedRangeInvalid)
        throw IllegalArgumentException("Cannot index tensor with shape: ${this} with index: ${Arrays.toString(index)}")

    return normalizedShapeIndexMapping.map { it.second }.toTypedArray()
}

fun List<Int>.normalizedIndexedShape(index: Array<out IntRange>): Pair<Array<IntRange>, List<Int>> {
    val normalizedIndex = normalizeIndex(index)

    return Pair(
        normalizedIndex,
        normalizedIndex.map { it.last - it.first + 1 } + this.drop(normalizedIndex.size)
    )
}

fun List<Int>.indexSelectNormAxisShape(axis: Int, indexShape: List<Int>): Pair<Int, List<Int>> {
    val normAxis = normalizeAxis(axis)
    if (indexShape.size != 1) {
        throw IllegalArgumentException("Index select requires singleton tensor as index, but got index tensor with shape: $indexShape")
    }
    val resShape = toMutableList()
    resShape[normAxis] = indexShape[0]
    return normAxis to resShape
}

fun List<Int>.indexAddCheckShapesNormAxis(axis: Int, indexShape: List<Int>, srcShape: List<Int>): Int {
    if (indexShape.size != 1) {
        throw IllegalArgumentException("Index add requires index tensor to have 1 dimension, but got tensor with shape: $indexShape")
    }
    val normAxis = normalizeAxis(axis)
    val expectedSrcShape = toMutableList()
    expectedSrcShape[normAxis] = indexShape[0]
    if (srcShape != expectedSrcShape) {
        throw IllegalArgumentException("Index add expects source tensor to have same size as index size at the specified axis, " +
                "and same size as this tensor for every other axis, " +
                "but got source tensor with shape: $srcShape axis: $axis and this tensor have shape: $this")
    }
    return normAxis
}

fun List<Int>.tensorIndexing(index: IntArray, normalize: Boolean = true): SimpleTensorIndexing {
    val normalizedIndex = if (normalize) normalizeIndex(index) else index
    return SimpleTensorIndexing(this, normalizedIndex)
}

fun List<Int>.tensorIndexing(index: List<Int>, normalize: Boolean = true): SimpleTensorIndexing {
    val normalizedIndex = if (normalize) normalizeIndex(index) else index
    return SimpleTensorIndexing(this, normalizedIndex)
}

fun List<Int>.tensorIndexing(index: Array<out IntRange>, normalize: Boolean = true): RangeTensorIndexing {
    val normalizedIndex = if (normalize) normalizeIndex(index) else index
    return RangeTensorIndexing(this, normalizedIndex.toList())
}

fun List<Int>.toIndexRanges(): Array<IntRange> {
    return this.map { 0 until it }.toTypedArray()
}

private fun wrongShapeForOp(operationName: String, shape: List<Int>, operandName: String) {
    throw IllegalArgumentException("$operationName got tensor with shape: $shape as $operandName")
}

private val MV_OPERATION_NAME = "Matrix-vector multiplication"

fun List<Int>.mvShape(vectorShape: List<Int>): List<Int> {
    if (this.size != 2)
        wrongShapeForOp(MV_OPERATION_NAME, this, "matrix")
    if (vectorShape.size != 1)
        wrongShapeForOp(MV_OPERATION_NAME, vectorShape, "vector")
    if (this[1] != vectorShape[0])
        throw IllegalArgumentException(
            "Matrix-vector multiplication requires matrix tensor size at axis=1 as vector tensor at axis=0, " +
                    "but got matrix with shape: ${this} and vector with shape: ${vectorShape} (${this[1]} != ${vectorShape[0]}"
        )

    return listOf(this[0])
}

private val MM_OPERATION_NAME = "Matrix-matrix multiplication"

fun List<Int>.mmShape(otherShape: List<Int>): List<Int> {
    if (this.size != 2)
        wrongShapeForOp(MM_OPERATION_NAME, this, "first matrix")
    if (otherShape.size != 2)
        wrongShapeForOp(MM_OPERATION_NAME, otherShape, "second matrix")
    return listOf(this[0], otherShape[1])
}

fun List<Int>.toStoreSize(): Int = this.fold(1, Int::times)

fun List<Int>.normalizedAxisScatter(axis: Int, indexShape: List<Int>, srcShape: List<Int>): Int {
    if (this.size != indexShape.size || this.size != srcShape.size)
        throw IllegalArgumentException(
            "Scatter expects tensors (target, index, source) to have the same dimensions, " +
                    "but got tensors with shapes: target: ${this}, index: ${indexShape}, source: ${srcShape}"
        )

    val normalizedAxis = normalizeAxis(axis)

    for (a in this.indices) {
        if (indexShape[a] > srcShape[a])
            throw IllegalArgumentException(
                "Scatter requires that index.shape[a] <= src.shape[a] for all axis 'a', " +
                        "but got tensors with shapes: index: ${indexShape}, src: ${srcShape}"
            )

        if (a != normalizedAxis && indexShape[a] > this[a])
            throw IllegalArgumentException(
                "Scatter requires that index.shape[a] <= target.shape[a] for all axis 'a', " +
                        "except the specified 'axis' of the operation, " +
                        "but got tensors with shapes: index: ${indexShape}, target: ${this}"
            )
    }

    return normalizedAxis
}

fun List<Int>.dotShape(bShape: List<Int>): List<Int> {
    if (this.size != 1 || this != bShape)
        throw IllegalArgumentException(
            "Dot product can only applied to vectors (1D tensors) with same shape, " +
                    "but got tensors with shapes: $this and $bShape"
        )
    return listOf(1)
}

fun List<Int>.checkBlasShape(shape1: List<Int>, shape2: List<Int>) {
    if (shape1.size == 2 && shape2.size == 2) { //gemm
        if (this.size != 2 || this[0] != shape1[0] || this[1] != shape2[1])
            throw IllegalArgumentException("Gemm got tensors with invalid shapes, addTensor: $this , tensor1: $shape1 , tensor2: $shape2")
    } else if (shape1.size == 3 && shape2.size == 3) { //batchedGemm
        if (this.size != 3 || this[0] != shape1[0] || this[0] != shape2[0] || this[1] != shape1[1] || this[2] != shape2[2])
            throw IllegalArgumentException("Batched Gemm got tensors with invalid shapes, addTensor: $this , tensor1: $shape1 , tensor2: $shape2")
    } else if (shape1.size == 2 && shape2.size == 1) { //gemv
        if (this.size != 1 || this[0] != shape1[0] || shape1[1] != shape2[0])
            throw IllegalArgumentException("Gemv got tensors with invalid shapes, addTensor: $this , tensor1: $shape1 , tensor2: $shape2")
    } else {
        throw IllegalArgumentException("Invalid shapes in BLAS operation, addTensor: $this , tensor1: $shape1 , tensor2: $shape2")
    }
}

fun List<Int>.checkMatmulShapesCompatible(shape: List<Int>) {
    if (this.size > 1 && shape.size == 1 && this[this.size - 1] != shape[0]) {
        throw IllegalArgumentException("Cannot matrix multiply tensors with shape $this and $shape, shape sizes in the last dimension must match")
    }
    if (shape.size > 1 && this[this.size - 1] != shape[shape.size - 2]) {
        throw IllegalArgumentException("Cannot matrix multiply tensors with shape $this and $shape, the first shape size in the last dimension " +
                " must match the second shape size in the second to last dimension " +
                "(first.shape[${this.size - 1}] != second.shape[${shape.size - 2}])")
    }
}

