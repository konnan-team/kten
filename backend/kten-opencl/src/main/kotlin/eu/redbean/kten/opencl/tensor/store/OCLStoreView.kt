package eu.redbean.kten.opencl.tensor.store

import org.jocl.cl_mem

class OCLStoreView(
    private val range: IntRange,
    private val storeReference: OCLMemoryObject
) {

    val size
        get() = range.last - range.first + 1

    val offset: Long
        get() = range.first.toLong()

    fun getMemoryObject(accessOption: OCLMemoryObject.MemoryAccessOption): cl_mem {
        return storeReference.getMemoryObject(accessOption)
    }

    fun fill(value: Float) {
        storeReference.fill(value, range)
    }

}