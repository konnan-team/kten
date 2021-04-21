package eu.redbean.kten.api.tensor.store

import eu.redbean.kten.api.autograd.utils.toStoreSize
import java.util.stream.IntStream


private fun calcIndexBase(index: List<Int>, shape: List<Int>): Int {
    return index.foldIndexed(0) { idx, a, indexItem ->
        a * shape[idx] + indexItem
    }
}

abstract class AbstractIndexing<T>(
    protected val shape: List<Int>
) {
    abstract val sparseStoreRanges: List<IntRange>
    abstract val storeRange: IntRange

    abstract val tensorIndex: List<T>

    open val storeIndex: Int
        get() {
            val indexRange = storeRange
            if (indexRange.first != indexRange.last)
                throw IllegalArgumentException("Cannot access single value from tensor with shape: ${shape} at index: ${tensorIndex}")
            return indexRange.first
        }

    protected fun mapToSingleIndexBase(index: List<Int>): Int {
        return calcIndexBase(index, shape)
    }
}

class SimpleTensorIndexing(
    shape: List<Int>,
    override val tensorIndex: List<Int>
) : AbstractIndexing<Int>(shape) {

    constructor(shape: List<Int>, index: IntArray) : this(shape, index.toList())

    override val sparseStoreRanges: List<IntRange>
        get() = listOf(storeRange)

    override val storeRange: IntRange
        get() {
            val start = mapToSingleIndexBase(tensorIndex)

            if (tensorIndex.size == shape.size)
                return start..start

            val subDimVolume = shape.drop(tensorIndex.size).toStoreSize()

            return start * subDimVolume until (start + 1) * subDimVolume
        }

}

class RangeTensorIndexing(
    shape: List<Int>,
    override val tensorIndex: List<IntRange>
) : AbstractIndexing<IntRange>(shape) {

    constructor(shape: List<Int>, index: Array<IntRange>) : this(shape, index.toList())

    override val sparseStoreRanges: List<IntRange>
        get() {
            val sparseRanges = shape.zip(tensorIndex)
                .dropLastWhile { (max, range) -> range.first == 0 && range.last == max - 1 }
                .map { it.second }

            if (sparseRanges.isEmpty())
                return listOf(0 until shape.fold(1, Int::times))

            val subDimVolume = shape.drop(sparseRanges.size).toStoreSize()
            val sparseRangesShape = sparseRanges.map { it.last - it.first + 1 }
            val lastRange = sparseRanges.last()
            val sparseRangesShapeWithoutLast = sparseRangesShape.dropLast(1)
            val storeIndexing = StoreIndexing(sparseRangesShapeWithoutLast)
            val numberOfRanges = sparseRangesShapeWithoutLast.toStoreSize()

            return List(numberOfRanges) {
                val shiftedIndexForResultRange = storeIndexing.toTensorIndex(it) { dim -> sparseRanges[dim].first }.toList()
                val currentRangeFirstSingleIndex = shiftedIndexForResultRange + listOf(lastRange.first)
                val currentRangeLastSingleIndex = shiftedIndexForResultRange + listOf(lastRange.last + 1)

                mapToSingleIndexBase(currentRangeFirstSingleIndex) * subDimVolume until mapToSingleIndexBase(currentRangeLastSingleIndex) * subDimVolume
            } // TODO test me!!! (It needs separate test cases, it's too complicated to just test in the tensor uses)
        }

    override val storeRange: IntRange
        get() {
            val ranges = sparseStoreRanges
            if (ranges.size > 1)
                throw IllegalStateException("Range index: ${tensorIndex} does not reference a single store range in tensor with shape: ${shape}")
            return ranges.first()
        }
}


class StoreIndexing(
    private val shape: List<Int>
) {

    private val reversedShape = shape.reversed()
    private val size = shape.size

    val strides by lazy {
        shape.indices.map { if (it == shape.size - 1) 0 else shape.drop(it + 1).toStoreSize() }
    }

    fun toTensorIndex(storeIndex: Int, dimensionOffset: (Int) -> Int = { 0 }): IntArray {
        val indices = IntArray(size)
        var i = storeIndex
        var dimensionIndex = size - 1
        for (dimension in reversedShape) {
            indices[dimensionIndex] = (i % dimension) + dimensionOffset(dimensionIndex)
            i /= dimension
            dimensionIndex--
        }
        return indices
    }

    fun applyAt(axis: Int, operation: (indexMapping: (Int, StoreIndexing) -> Int) -> Unit) {
        val shapeWithoutDimAtAxis = shape.toMutableList()
        shapeWithoutDimAtAxis.removeAt(axis)
        val reversedShapeWithoutDimAtAxis = shapeWithoutDimAtAxis.asReversed()
        val elements = shapeWithoutDimAtAxis.toStoreSize()

        IntStream.range(0, elements)
            .forEach {
                val indices = MutableList(shapeWithoutDimAtAxis.size) { 0 }
                var i = it
                var dimensionIndex = shapeWithoutDimAtAxis.size - 1
                for (dimension in reversedShapeWithoutDimAtAxis) {
                    indices[dimensionIndex] = (i % dimension)
                    i /= dimension
                    dimensionIndex--
                }
                indices.add(axis, 0)
                operation { index, storeIndexing ->
                    indices[axis] = index
                    storeIndexing.indexBase(indices)
                }
            }
    }

    fun indexBase(index: List<Int>): Int {
        return calcIndexBase(index, shape)
    }

}