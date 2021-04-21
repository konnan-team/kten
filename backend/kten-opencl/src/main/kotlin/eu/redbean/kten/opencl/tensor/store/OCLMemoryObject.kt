package eu.redbean.kten.opencl.tensor.store

import eu.redbean.kten.opencl.tensor.platform.OCLEnvironment
import eu.redbean.kten.opencl.tensor.platform.OCLPlatformInitializer
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.SynchronizationLevel.*
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.MemoryAccessOption.*
import org.jocl.CL
import org.jocl.Pointer
import org.jocl.Sizeof
import org.jocl.blast.CLBlast
import org.jocl.cl_mem
import java.util.*

class OCLMemoryObject {

    enum class SynchronizationLevel {
        OFF_DEVICE, ON_DEVICE, BOTH
    }

    enum class MemoryAccessOption {
        SOURCE, TARGET, SOURCE_AND_TARGET
    }

    val environment: OCLEnvironment

    val size: Int

    val memSize: Long
        get() = size.toLong() * Sizeof.cl_float

    private val memObject: cl_mem

    var synchronization: SynchronizationLevel

    private var jvmArrayBacking: FloatArray? = null
    private var pointerBacking: Pointer? = null

    val jvmArray: FloatArray
        get() {
            if (jvmArrayBacking == null) {
                jvmArrayBacking = FloatArray(size)
                this.pointerBacking = Pointer.to(jvmArrayBacking)
            }
            return jvmArrayBacking!!
        }

    val pointer: Pointer
        get() {
            if (pointerBacking == null) {
                this.jvmArrayBacking = FloatArray(size)
                pointerBacking = Pointer.to(this.jvmArrayBacking)
            }
            return pointerBacking!!
        }

    private var released = false
    private var reusable = false


    constructor(array: FloatArray, environment: OCLEnvironment, synchronization: SynchronizationLevel = OFF_DEVICE) {
        this.jvmArrayBacking = array
        this.size = array.size
        this.memObject = CL.clCreateBuffer(
            environment.context,
            CL.CL_MEM_READ_WRITE,
            Sizeof.cl_float * array.size.toLong(),
            null,
            null
        )
        this.pointerBacking = Pointer.to(array)
        this.environment = environment
        this.synchronization = synchronization
//        OCLPlatformInitializer.cleaner.register(this) {
//            if (!released)
//                CL.clReleaseMemObject(memObject)
//        }
    }

    constructor(size: Int, environment: OCLEnvironment, synchronization: SynchronizationLevel = ON_DEVICE) {
        this.size = size
        this.memObject = CL.clCreateBuffer(
            environment.context,
            CL.CL_MEM_READ_WRITE,
            Sizeof.cl_float * size.toLong(),
            null,
            null
        )
        this.environment = environment
        this.synchronization = synchronization
//        OCLPlatformInitializer.cleaner.register(this) {
//            if (!released)
//                CL.clReleaseMemObject(memObject)
//        }
    }

    fun makeReusable() {
        jvmArrayBacking = null
        pointerBacking = null
        synchronization = ON_DEVICE
        reusable = true
    }

    fun reuseWithJvmArray(array: FloatArray, synchronization: SynchronizationLevel) {
        jvmArrayBacking = array
        pointerBacking = Pointer.to(array)
        this.synchronization = synchronization
        reusable = false
    }

    fun reuseWithoutJvmBacking(synchronization: SynchronizationLevel) {
        this.synchronization = synchronization
        reusable = false
    }

    fun isReusable(): Boolean {
        return reusable && !released
    }

    fun release() {
        CL.clReleaseMemObject(memObject)
        released = true
    }

    fun getMemoryObject(option: MemoryAccessOption): cl_mem {
        if (option == TARGET) {
            synchronization = ON_DEVICE
            return memObject
        }

        if (synchronization == OFF_DEVICE) {
            writeToDevice()
        }

        if (option == SOURCE_AND_TARGET) {
            synchronization = ON_DEVICE
        }

        return memObject
    }

    fun readToArray() {
        if (synchronization == OFF_DEVICE || synchronization == BOTH) {
            return
        }

        CL.clEnqueueReadBuffer(
            environment.commandQueue,
            memObject,
            CL.CL_TRUE,
            0,
            memSize,
            pointer,
            0,
            null,
            null
        )

        synchronization = BOTH
    }

    fun writeToDevice() {
        if (synchronization == ON_DEVICE || synchronization == BOTH) {
            return
        }

        CL.clEnqueueWriteBuffer(
            environment.commandQueue,
            memObject,
            CL.CL_TRUE,
            0,
            memSize,
            pointer,
            0,
            null,
            null
        )

        synchronization = BOTH
    }


    fun copyOf(): OCLMemoryObject {
        if (synchronization == OFF_DEVICE) {
            return OCLMemoryObject(jvmArray.copyOf(), environment)
        }

        val res = OCLMemoryObject(size, environment)

        CLBlast.CLBlastScopy(
            size.toLong(),
            memObject,
            0L,
            1L,
            res.getMemoryObject(TARGET),
            0L,
            1L,
            environment.commandQueue,
            null
        )

        return res
    }

    fun slice(range: IntRange): OCLMemoryObject {
        if (synchronization == OFF_DEVICE) {
            return OCLMemoryObject(jvmArray.sliceArray(range), environment)
        }

        val newSize = range.last - range.first + 1

        val res = OCLMemoryObject(newSize, environment)

        CLBlast.CLBlastScopy(
            newSize.toLong(),
            memObject,
            range.first.toLong(),
            1L,
            res.getMemoryObject(TARGET),
            0L,
            1L,
            environment.commandQueue,
            null
        )

        return res
    }

    fun sparseSlices(ranges: List<IntRange>): OCLMemoryObject {
        if (synchronization == OFF_DEVICE) {
            return OCLMemoryObject(ranges.map { jvmArray.sliceArray(it) }.reduce(FloatArray::plus), environment)
        }

        val sizes = ranges.map { it.last - it.first + 1 }
        val newSize = sizes.sum()

        val res = OCLMemoryObject(newSize, environment)

        var targetOffset = 0L
        for (i in ranges.indices) {
            CLBlast.CLBlastScopy(
                sizes[i].toLong(),
                memObject,
                ranges[i].first.toLong(),
                1L,
                res.getMemoryObject(TARGET),
                targetOffset,
                1L,
                environment.commandQueue,
                null
            )
            targetOffset += sizes[i].toLong()
        }

        return res
    }

    fun copyIntoAt(index: Int, other: OCLMemoryObject) {
        CLBlast.CLBlastScopy(
            other.size.toLong(),
            other.getMemoryObject(SOURCE),
            0L,
            1L,
            this.getMemoryObject(SOURCE_AND_TARGET),
            index.toLong(),
            1L,
            environment.commandQueue,
            null
        )
    }

    fun copyIntoSparseRanges(ranges: List<IntRange>, other: OCLMemoryObject) {
        val sizes = ranges.map { it.last - it.first + 1 }
        var sourceStart = 0L
        for (i in ranges.indices) {
            CLBlast.CLBlastScopy(
                sizes[i].toLong(),
                other.getMemoryObject(SOURCE),
                sourceStart,
                1L,
                this.getMemoryObject(SOURCE_AND_TARGET),
                ranges[i].first.toLong(),
                1L,
                environment.commandQueue,
                null
            )
            sourceStart += sizes[i].toLong()
        }
    }

    override fun equals(other: Any?): Boolean {
        if (other !is OCLMemoryObject)
            return false

        if (this.size != other.size)
            return false

        this.readToArray()
        other.readToArray()

        return this.jvmArray.contentEquals(other.jvmArray)
    }

    override fun hashCode(): Int {
        if (synchronization == OFF_DEVICE)
            return Objects.hash(environment, size, jvmArray)
        return Objects.hash(environment, size, memObject)
    }

    operator fun get(index: Int): Float {
        readToArray()
        return jvmArray[index]
    }

    operator fun set(index: Int, value: Float) {
        if (synchronization == OFF_DEVICE || synchronization == BOTH)
            jvmArray[index] = value

        val wasBoth = (synchronization == BOTH)

        if (synchronization == ON_DEVICE || synchronization == BOTH)
            environment.kernelStore.fill(this, index..index, value)

        if (wasBoth)
            synchronization = BOTH
    }

    fun fill(value: Float) {
        if (synchronization == OFF_DEVICE || synchronization == BOTH)
            jvmArray.fill(value)

        val wasBoth = (synchronization == BOTH)

        if (synchronization == ON_DEVICE || synchronization == BOTH)
            environment.kernelStore.fill(this, 0 until size, value)

        if (wasBoth)
            synchronization = BOTH
    }

    fun fill(value: Float, range: IntRange) {
        if (synchronization == OFF_DEVICE || synchronization == BOTH)
            jvmArray.fill(value, range.first, range.last + 1)

        val wasBoth = (synchronization == BOTH)

        if (synchronization == ON_DEVICE || synchronization == BOTH)
            environment.kernelStore.fill(this, range, value)

        if (wasBoth)
            synchronization = BOTH
    }

    fun containsNan(): Boolean {
        if (synchronization == OFF_DEVICE)
            return jvmArray.any { it.isNaN() }
        return environment.kernelStore.containsNaN(this)
    }
}