package eu.redbean.kten.jvm.tensor.store

class StoreView(
    private val range: IntRange,
    private val storeRef: FloatArray
) {

    val size
        get() = range.last - range.first + 1

    operator fun get(index: Int): Float {
        if (range.contains(range.first + index).not())
            throw IndexOutOfBoundsException("Index: $index is out of bounds for range: $range")
        return storeRef[range.first + index]
    }

    operator fun set(index: Int, value: Float) {
        if (range.contains(range.first + index).not())
            throw IndexOutOfBoundsException()
        storeRef[range.first + index] = value
    }

    fun fill(value: Float) {
        storeRef.fill(value, range.first, range.last + 1)
    }

}