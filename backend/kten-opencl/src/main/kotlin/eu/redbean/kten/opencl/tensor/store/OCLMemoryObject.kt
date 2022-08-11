package eu.redbean.kten.opencl.tensor.store

import eu.redbean.kten.opencl.tensor.platform.OCLEnvironment
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.MemoryAccessOption.*
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.SynchronizationLevel.*
import org.jocl.CL
import org.jocl.Pointer
import org.jocl.Sizeof
import org.jocl.blast.CLBlast
import org.jocl.cl_mem

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

    private var memObjectBacking: cl_mem? = null

    private val memObject: cl_mem
        get() {
            if (memObjectBacking == null) {
                memObjectBacking = CL.clCreateBuffer(
                    environment.context,
                    CL.CL_MEM_READ_WRITE,
                    Sizeof.cl_float * size.toLong(),
                    null,
                    null
                )
                environment.registerActiveMemObject(this)
            }
            return memObjectBacking!!
        }

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

    val reusable get() = refCount <= 0
    private var released = false

    private var refCount = 1

    var lastAccess = System.nanoTime()
        get private set


    constructor(array: FloatArray, environment: OCLEnvironment, synchronization: SynchronizationLevel = OFF_DEVICE) {
        this.jvmArrayBacking = array
        this.size = array.size
        this.memObjectBacking = CL.clCreateBuffer(
            environment.context,
            CL.CL_MEM_READ_WRITE,
            Sizeof.cl_float * array.size.toLong(),
            null,
            null
        )
        this.pointerBacking = Pointer.to(array)
        this.environment = environment
        this.synchronization = synchronization
        environment.registerActiveMemObject(this)
    }

    constructor(size: Int, environment: OCLEnvironment, synchronization: SynchronizationLevel = ON_DEVICE) {
        this.size = size
        this.memObjectBacking = CL.clCreateBuffer(
            environment.context,
            CL.CL_MEM_READ_WRITE,
            Sizeof.cl_float * size.toLong(),
            null,
            null
        )
        this.environment = environment
        this.synchronization = synchronization
        environment.registerActiveMemObject(this)
    }

    fun incrementRef() {
        if (released || reusable) {
            throw IllegalStateException("Store reference is already marked for reuse or release.")
        }
        refCount++
    }

    fun makeReusable() {
        refCount--
        if (refCount < 0) {
            throw IllegalStateException("Ref count is less than 0")
        }
        if (reusable) {
            jvmArrayBacking = null
            pointerBacking = null
            synchronization = ON_DEVICE
        }
    }

    fun reuseWithJvmArray(array: FloatArray, synchronization: SynchronizationLevel) {
        if (reusable.not()) {
            throw IllegalStateException("Trying to reuse store reference, which is not reusable.")
        }
        refCount = 1
        lastAccess = System.nanoTime()
        jvmArrayBacking = array
        pointerBacking = Pointer.to(array)
        this.synchronization = synchronization
    }

    fun reuseWithoutJvmBacking(synchronization: SynchronizationLevel) {
        if (reusable.not()) {
            throw IllegalStateException("Trying to reuse store reference, which is not reusable.")
        }
        refCount = 1
        lastAccess = System.nanoTime()
        this.synchronization = synchronization
    }

    fun isReusable(): Boolean {
        return reusable && !released
    }

    fun isReleased(): Boolean = released

    fun release() {
        memObjectBacking?.let {
            synchronized(it) {
                if (!released) {
                    CL.clReleaseMemObject(it)
                    released = true
                }
            }
        }
        jvmArrayBacking = null
        released = true
        environment.removeInactiveMemObject(this)
    }

    fun getMemoryObject(option: MemoryAccessOption): cl_mem {
        if (reusable || released) {
            throw IllegalStateException("Reusable memory object access violation")
        }

        lastAccess = System.nanoTime()
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
        lastAccess = System.nanoTime()
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

        jvmArrayBacking = null
        pointerBacking = null

        synchronization = ON_DEVICE
    }


    fun copyOf(): OCLMemoryObject {
        lastAccess = System.nanoTime()
        if (synchronization == OFF_DEVICE) {
            return environment.memoryObjectOf(jvmArray.copyOf())
        }

        val res = environment.memoryObject(size)

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
        lastAccess = System.nanoTime()
        if (synchronization == OFF_DEVICE) {
            return environment.memoryObjectOf(jvmArray.sliceArray(range))
        }

        val newSize = range.last - range.first + 1

        val res = environment.memoryObject(newSize)

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
        lastAccess = System.nanoTime()
        if (synchronization == OFF_DEVICE) {
            return environment.memoryObjectOf(ranges.map { jvmArray.sliceArray(it) }.reduce(FloatArray::plus))
        }

        val sizes = ranges.map { it.last - it.first + 1 }
        val newSize = sizes.sum()

        val res = environment.memoryObject(newSize)

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
        lastAccess = System.nanoTime()
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
        System.nanoTime()
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

        return this.hashCode() == other.hashCode()
    }

    operator fun get(index: Int): Float {
        lastAccess = System.nanoTime()
        readToArray()
        return jvmArray[index]
    }

    operator fun set(index: Int, value: Float) {
        lastAccess = System.nanoTime()
        if (synchronization == OFF_DEVICE || synchronization == BOTH)
            jvmArray[index] = value

        val wasBoth = (synchronization == BOTH)

        if (synchronization == ON_DEVICE || synchronization == BOTH)
            environment.kernelStore.fill(this, index..index, value)

        if (wasBoth)
            synchronization = BOTH
    }

    fun fill(value: Float) {
        lastAccess = System.nanoTime()
        if (synchronization == OFF_DEVICE || synchronization == BOTH)
            jvmArray.fill(value)

        val wasBoth = (synchronization == BOTH)

        if (synchronization == ON_DEVICE || synchronization == BOTH)
            environment.kernelStore.fill(this, 0 until size, value)

        if (wasBoth)
            synchronization = BOTH
    }

    fun fill(value: Float, range: IntRange) {
        lastAccess = System.nanoTime()
        if (synchronization == OFF_DEVICE || synchronization == BOTH)
            jvmArray.fill(value, range.first, range.last + 1)

        val wasBoth = (synchronization == BOTH)

        if (synchronization == ON_DEVICE || synchronization == BOTH)
            environment.kernelStore.fill(this, range, value)

        if (wasBoth)
            synchronization = BOTH
    }

    fun containsNan(): Boolean {
        lastAccess = System.nanoTime()
        if (synchronization == OFF_DEVICE)
            return jvmArray.any { it.isNaN() }
        return environment.kernelStore.containsNaN(this)
    }

    fun manageUnusedDeviceMemory(immediatelly: Boolean = false): Boolean {
        val now = System.nanoTime()
        if (immediatelly || lastAccess < now - 5_000_000_000L) {
            if (!isReleased()) {
                if (!isReusable()) {
                    readToArray()
                }
                memObjectBacking?.let {
                    synchronized(it) {
                        CL.clReleaseMemObject(it)
                    }
                }
                memObjectBacking = null
                synchronization = OFF_DEVICE
            }
            return true
        }
        return false
    }

}